"""
api/main.py — FastAPI gateway for the Real-Time Digital Twin.

Endpoints
---------
GET  /state                — current warehouse state (JSON)
GET  /state/bev            — BEV overhead map (JPEG image)
GET  /incidents            — recent active incidents (JSON)
WS   /stream               — WebSocket: real-time state stream (JSON)
GET  /analytics/throughput — KPI metrics (JSON)
GET  /health               — liveness probe

Run with::

    uvicorn warehousegpt.digital_twin.api.main:app \
        --host 0.0.0.0 --port 8100 --reload

Environment variables (see Settings)
-------------------------------------
DT_REDIS_URL        redis://localhost:6379
DT_PG_DSN           postgresql://user:pass@localhost/warehouse
DT_WAREHOUSE_ID     default
DT_ANALYTICS_WINDOW 3600   (seconds)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from warehousegpt.digital_twin.analytics.throughput import (
    AnalyticsConfig,
    ThroughputAnalytics,
)
from warehousegpt.digital_twin.state_estimation.warehouse_state import (
    AgentPose,
    Incident,
    InventoryLocation,
    WarehouseState,
    WarehouseStateEstimator,
)
from warehousegpt.digital_twin.storage.event_store import EventStore, WarehouseEvent
from warehousegpt.digital_twin.storage.state_store import StateStore
from warehousegpt.digital_twin.safety.pipeline import SafetyPipeline
from warehousegpt.digital_twin.world_model_sync import WorldModelSynchronizer
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

_agents_gauge = Gauge(
    "warehouse_agents_total",
    "Number of tracked agents currently in the warehouse",
    ["agent_type"],
)
_incidents_counter = Counter(
    "warehouse_incidents_total",
    "Total safety incidents detected",
    ["incident_type", "severity"],
)
_inject_requests = Counter(
    "warehouse_inject_requests_total",
    "Total /inject calls received",
)
_inject_latency = Histogram(
    "warehouse_inject_latency_seconds",
    "Processing latency of /inject endpoint",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
_camera_frames = Counter(
    "warehouse_camera_frames_total",
    "Total camera frames processed via /inject/camera",
    ["camera_id"],
)
_fires_counter = Counter(
    "warehouse_fires_detected_total",
    "Total fire events detected from camera frames",
    ["camera_id"],
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DT_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379"
    pg_dsn: str = "postgresql://warehousegpt:secret@localhost:5432/warehouse"
    warehouse_id: str = "default"
    analytics_window: float = 3600.0
    bev_jpeg_quality: int = 85
    stream_max_clients: int = 50
    stream_send_interval: float = 0.1   # seconds between WS pushes (≈10 Hz)


settings = Settings()


# ---------------------------------------------------------------------------
# Application state (singletons, shared across requests)
# ---------------------------------------------------------------------------


class AppState:
    """Holds all singleton services wired at startup."""

    def __init__(self) -> None:
        self.state_store: StateStore | None = None
        self.event_store: EventStore | None = None
        self.estimator: WarehouseStateEstimator = WarehouseStateEstimator()
        self.analytics: ThroughputAnalytics = ThroughputAnalytics(
            config=AnalyticsConfig(window_seconds=settings.analytics_window)
        )
        self.sync: WorldModelSynchronizer = WorldModelSynchronizer()
        self.safety: SafetyPipeline = SafetyPipeline()
        self._ws_clients: set["WebSocket"] = set()
        self._latest_state: WarehouseState | None = None
        self._bev_cache: bytes | None = None   # JPEG bytes

    def get_latest_state(self) -> WarehouseState | None:
        return self._latest_state

    def set_latest_state(self, state: WarehouseState) -> None:
        self._latest_state = state
        self.analytics.ingest(state)

    def add_ws_client(self, ws: "WebSocket") -> bool:
        if len(self._ws_clients) >= settings.stream_max_clients:
            return False
        self._ws_clients.add(ws)
        return True

    def remove_ws_client(self, ws: "WebSocket") -> None:
        self._ws_clients.discard(ws)

    async def broadcast(self, payload: str) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._ws_clients.discard(ws)

    @property
    def ws_client_count(self) -> int:
        return len(self._ws_clients)


_app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start background services on startup; clean up on shutdown."""
    logger.info("Digital Twin API starting …")

    # Connect to Redis
    _app_state.state_store = StateStore(
        redis_url=settings.redis_url,
        warehouse_id=settings.warehouse_id,
    )
    try:
        await _app_state.state_store.connect()
    except Exception as exc:
        logger.warning("Redis unavailable: %s — state store disabled.", exc)
        _app_state.state_store = None

    # Connect to PostgreSQL / TimescaleDB
    _app_state.event_store = EventStore(
        dsn=settings.pg_dsn,
        warehouse_id=settings.warehouse_id,
    )
    try:
        await _app_state.event_store.connect()
    except Exception as exc:
        logger.warning("PostgreSQL unavailable: %s — event store disabled.", exc)
        _app_state.event_store = None

    # Wire world-model synchronizer output → app state
    _app_state.sync.on_blended_state(_on_new_state)

    # Start background tasks
    bg_tasks = [
        asyncio.create_task(_app_state.sync.run(), name="world-model-sync"),
        asyncio.create_task(_broadcaster_loop(), name="ws-broadcaster"),
    ]

    if _app_state.state_store:
        bg_tasks.append(
            asyncio.create_task(
                _redis_subscriber_loop(), name="redis-state-sub"
            )
        )

    logger.info("Digital Twin API ready.")
    yield

    # Shutdown
    logger.info("Digital Twin API shutting down …")
    for task in bg_tasks:
        task.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)

    await _app_state.sync.stop()

    if _app_state.state_store:
        await _app_state.state_store.close()
    if _app_state.event_store:
        await _app_state.event_store.close()

    logger.info("Digital Twin API stopped.")


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def _on_new_state(state: WarehouseState) -> None:
    """
    Callback from WorldModelSynchronizer — called each time a blended state
    is produced.  Persists to Redis and updates in-memory cache.
    """
    _app_state.set_latest_state(state)

    if _app_state.state_store:
        try:
            await _app_state.state_store.set_state(state)
        except Exception as exc:
            logger.warning("Redis write error: %s", exc)

    # Render BEV and cache as JPEG
    bev = _app_state.estimator.get_bev_map()
    _app_state._bev_cache = _encode_jpeg(bev, settings.bev_jpeg_quality)


async def _broadcaster_loop() -> None:
    """Push the latest state to all connected WebSocket clients at ~10 Hz."""
    while True:
        try:
            state = _app_state.get_latest_state()
            if state and _app_state.ws_client_count > 0:
                payload = json.dumps(state.as_dict())
                await _app_state.broadcast(payload)
        except Exception as exc:
            logger.warning("Broadcaster error: %s", exc)
        await asyncio.sleep(settings.stream_send_interval)


async def _redis_subscriber_loop() -> None:
    """
    Subscribe to Redis state-update channel and push blended states
    back into the synchronizer when running in distributed mode
    (i.e. the estimator runs on a separate process).
    """
    if _app_state.state_store is None:
        return
    try:
        async for state in _app_state.state_store.subscribe_state_updates():
            _app_state.sync.push_state(state)
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.error("Redis subscriber error: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="WarehouseGPT Digital Twin API",
    description=(
        "Real-time digital twin gateway providing warehouse state, "
        "BEV maps, incident feeds, and streaming WebSocket updates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Expose Prometheus metrics at GET /metrics
app.mount("/metrics", make_asgi_app())


# ---------------------------------------------------------------------------
# Inject models (used by Isaac Sim HTTP bridge)
# ---------------------------------------------------------------------------


class AgentPoseIn(BaseModel):
    agent_id: str
    agent_type: str
    x: float
    y: float
    z: float = 0.0
    heading_rad: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 1.0


class PalletIn(BaseModel):
    item_id: str
    barcode: str = ""
    x: float
    y: float
    z: float = 0.0
    zone: str = "unknown"


class IncidentIn(BaseModel):
    incident_id: str
    incident_type: str
    severity: str = "medium"
    description: str = ""
    agent_ids: list[str] = []
    location_x: float = 0.0
    location_y: float = 0.0


class InjectRequest(BaseModel):
    forklifts: list[AgentPoseIn] = []
    workers: list[AgentPoseIn] = []
    amrs: list[AgentPoseIn] = []
    pallets: list[PalletIn] = []
    incidents: list[IncidentIn] = []
    frame_index: int = 0


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["infra"])
async def health() -> dict[str, Any]:
    """Liveness probe."""
    return {
        "status": "ok",
        "warehouse_id": settings.warehouse_id,
        "timestamp": time.time(),
        "ws_clients": _app_state.ws_client_count,
        "redis_connected": _app_state.state_store is not None,
        "pg_connected": _app_state.event_store is not None,
    }


@app.get("/bev/live", response_class=HTMLResponse, tags=["state"])
async def bev_live_viewer() -> HTMLResponse:
    """Live auto-refreshing Bird's-Eye-View map (no manual refresh needed)."""
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>WarehouseGPT — Live BEV</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0f1117; color: #e0e0e0; font-family: monospace; display: flex; flex-direction: column; height: 100vh; }
    #header { background: #1a1d27; padding: 10px 16px; display: flex; align-items: center; gap: 20px; border-bottom: 1px solid #2a2d3a; }
    #header h1 { font-size: 15px; color: #7eb8f7; letter-spacing: 1px; }
    .stat { font-size: 12px; color: #888; }
    .stat span { color: #e0e0e0; }
    #dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; margin-left: auto; }
    #dot.dead { background: #ef4444; }
    #bev-wrap { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 8px; }
    #bev { max-width: 100%; max-height: 100%; border: 1px solid #2a2d3a; border-radius: 4px; }
    #legend { background: #1a1d27; padding: 8px 16px; display: flex; gap: 20px; border-top: 1px solid #2a2d3a; flex-wrap: wrap; }
    .leg { display: flex; align-items: center; gap: 6px; font-size: 11px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; }
  </style>
</head>
<body>
  <div id="header">
    <h1>WAREHOUSEGPT — LIVE BEV</h1>
    <div class="stat">FPS: <span id="fps">0</span></div>
    <div class="stat">Forklifts: <span id="fl">0</span></div>
    <div class="stat">Workers: <span id="wk">0</span></div>
    <div class="stat">AMRs: <span id="amr">0</span></div>
    <div class="stat">Incidents: <span id="inc">0</span></div>
    <div id="dot"></div>
  </div>
  <div id="bev-wrap">
    <img id="bev" src="/state/bev" alt="BEV map loading...">
  </div>
  <div id="legend">
    <div class="leg"><div class="dot" style="background:#00b4ff"></div> Forklift</div>
    <div class="leg"><div class="dot" style="background:#00ff64"></div> Worker</div>
    <div class="leg"><div class="dot" style="background:#ffdc00"></div> AMR</div>
    <div class="leg"><div class="dot" style="background:#a0a0a0;border-radius:2px"></div> Pallet</div>
    <div class="leg"><div class="dot" style="background:#ff3333"></div> Incident</div>
  </div>
  <script>
    const img = document.getElementById('bev');
    const dot = document.getElementById('dot');
    let lastT = performance.now(), frameCount = 0, fps = 0;

    async function refreshState() {
      try {
        const r = await fetch('/state');
        if (r.ok) {
          const s = await r.json();
          document.getElementById('fl').textContent  = (s.forklifts || []).length;
          document.getElementById('wk').textContent  = (s.workers   || []).length;
          document.getElementById('amr').textContent = (s.amrs      || []).length;
          document.getElementById('inc').textContent = (s.active_incidents || []).length;
          dot.className = '';
        }
      } catch { dot.className = 'dead'; }
    }

    function refreshBev() {
      const src = '/state/bev?t=' + Date.now();
      const tmp = new Image();
      tmp.onload = () => {
        img.src = tmp.src;
        frameCount++;
        const now = performance.now();
        if (now - lastT >= 1000) {
          fps = Math.round(frameCount * 1000 / (now - lastT));
          document.getElementById('fps').textContent = fps;
          frameCount = 0; lastT = now;
        }
      };
      tmp.src = src;
    }

    setInterval(refreshBev, 150);
    setInterval(refreshState, 500);
    refreshState();
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/inject", tags=["isaac-sim"])
async def inject_state_from_sim(payload: InjectRequest) -> dict[str, Any]:
    """
    Receive live warehouse state from Isaac Sim (or any external source).

    Called by the Isaac Sim Script Editor bridge every ~100 ms.
    Converts the payload into a WarehouseState, renders the BEV map,
    and pushes the state through the WorldModelSynchronizer.
    """
    _inject_requests.inc()
    _t0 = time.monotonic()

    forklifts = [
        AgentPose(
            agent_id=a.agent_id, agent_type=a.agent_type,
            x=a.x, y=a.y, z=a.z,
            heading_rad=a.heading_rad, vx=a.vx, vy=a.vy,
            confidence=a.confidence,
        )
        for a in payload.forklifts
    ]
    workers = [
        AgentPose(
            agent_id=a.agent_id, agent_type=a.agent_type,
            x=a.x, y=a.y, z=a.z,
            heading_rad=a.heading_rad, vx=a.vx, vy=a.vy,
            confidence=a.confidence,
        )
        for a in payload.workers
    ]
    amrs = [
        AgentPose(
            agent_id=a.agent_id, agent_type=a.agent_type,
            x=a.x, y=a.y, z=a.z,
            heading_rad=a.heading_rad, vx=a.vx, vy=a.vy,
            confidence=a.confidence,
        )
        for a in payload.amrs
    ]
    pallets = [
        InventoryLocation(
            item_id=p.item_id, barcode=p.barcode,
            x=p.x, y=p.y, z=p.z, zone=p.zone,
        )
        for p in payload.pallets
    ]
    incidents = [
        Incident(
            incident_id=inc.incident_id,
            incident_type=inc.incident_type,
            severity=inc.severity,
            timestamp=time.monotonic(),
            description=inc.description,
            agent_ids=inc.agent_ids,
            location_x=inc.location_x,
            location_y=inc.location_y,
        )
        for inc in payload.incidents
    ]

    # Render BEV and cache
    bev = _app_state.estimator._render_bev(forklifts, workers, amrs, pallets)
    _app_state.estimator._bev_cache = bev
    _app_state._bev_cache = _encode_jpeg(bev, settings.bev_jpeg_quality)

    # --- Safety AI pipeline -------------------------------------------------
    # bev_frame is NOT passed for fire detection: the BEV renders forklifts as
    # orange circles which the colour-based fire detector misclassifies.
    # Fire detection is reserved for real camera frames injected via /inject/camera.
    detected_incidents = _app_state.safety.analyze(
        forklifts=forklifts,
        workers=workers,
        amrs=amrs,
        bev_frame=None,
        frame_index=payload.frame_index,
    )

    # Merge: payload incidents take precedence; auto-detected ones are appended
    payload_ids = {inc.incident_id for inc in incidents}
    for inc in detected_incidents:
        if inc.incident_id not in payload_ids:
            incidents.append(inc)

    # Persist high/critical incidents to the event store
    if _app_state.event_store and detected_incidents:
        import datetime
        import asyncio

        async def _persist() -> None:
            for inc in detected_incidents:
                if inc.severity not in ("high", "critical"):
                    continue
                ev = WarehouseEvent(
                    event_type=inc.incident_type,
                    severity=inc.severity,
                    agent_ids=inc.agent_ids,
                    location_x=inc.location_x,
                    location_y=inc.location_y,
                    payload={"description": inc.description, "incident_id": inc.incident_id},
                    event_time=datetime.datetime.now(tz=datetime.timezone.utc),
                    warehouse_id=settings.warehouse_id,
                )
                try:
                    await _app_state.event_store.write_event(ev)
                except Exception as exc:
                    logger.warning("EventStore write error: %s", exc)

        asyncio.create_task(_persist())
    # ------------------------------------------------------------------------

    _app_state.estimator._active_incidents = incidents
    state = WarehouseState(
        timestamp=time.monotonic(),
        frame_index=payload.frame_index,
        forklift_poses=forklifts,
        worker_poses=workers,
        amr_poses=amrs,
        inventory_locations=pallets,
        active_incidents=incidents,
        track_count=len(forklifts) + len(workers) + len(amrs),
    )
    _app_state.set_latest_state(state)

    # Update Prometheus metrics
    _inject_latency.observe(time.monotonic() - _t0)
    _agents_gauge.labels(agent_type="forklift").set(len(forklifts))
    _agents_gauge.labels(agent_type="worker").set(len(workers))
    _agents_gauge.labels(agent_type="amr").set(len(amrs))
    for inc in detected_incidents:
        _incidents_counter.labels(
            incident_type=inc.incident_type, severity=inc.severity
        ).inc()

    return {
        "status": "ok",
        "frame_index": payload.frame_index,
        "agents": len(forklifts) + len(workers) + len(amrs),
        "safety_incidents_detected": len(detected_incidents),
    }


@app.post("/inject/camera", tags=["safety"])
async def inject_camera_frame(
    frame: UploadFile = File(..., description="Camera frame — JPEG or PNG, BGR colour order"),
    camera_id: str = Form("cam_0", description="Camera identifier"),
    frame_index: int = Form(0, description="Simulation frame counter"),
) -> dict[str, Any]:
    """
    Run fire detection on a real camera frame from Isaac Sim or a physical camera.

    Accepts a JPEG/PNG image upload, decodes it to BGR, and runs the FireDetector.
    Empty agent lists are passed to the safety pipeline so only fire detection runs
    (near-miss / collision / zone checks require world-coordinate poses from /inject).

    Any detected fires are:
    - Persisted to the event store (PostgreSQL)
    - Merged into the active-incident list so /state and /bev/live reflect them
    """
    import asyncio as _asyncio
    import datetime

    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image — send JPEG or PNG.")

    detected = _app_state.safety.analyze(
        forklifts=[],
        workers=[],
        amrs=[],
        bev_frame=bgr,
        frame_index=frame_index,
    )

    # Merge fires into current live state so /state and BEV reflect them immediately
    if detected:
        state = _app_state.get_latest_state()
        if state:
            existing_ids = {inc.incident_id for inc in state.active_incidents}
            for inc in detected:
                if inc.incident_id not in existing_ids:
                    state.active_incidents.append(inc)

    # Persist to event store
    if _app_state.event_store and detected:
        async def _persist_fire() -> None:
            for inc in detected:
                ev = WarehouseEvent(
                    event_type=inc.incident_type,
                    severity=inc.severity,
                    agent_ids=inc.agent_ids,
                    location_x=inc.location_x,
                    location_y=inc.location_y,
                    payload={
                        "description": inc.description,
                        "incident_id": inc.incident_id,
                        "camera_id": camera_id,
                    },
                    event_time=datetime.datetime.now(tz=datetime.timezone.utc),
                    warehouse_id=settings.warehouse_id,
                )
                try:
                    await _app_state.event_store.write_event(ev)
                except Exception as exc:
                    logger.warning("EventStore fire write error: %s", exc)

        _asyncio.create_task(_persist_fire())

    _camera_frames.labels(camera_id=camera_id).inc()
    for _ in detected:
        _fires_counter.labels(camera_id=camera_id).inc()

    return {
        "status": "ok",
        "camera_id": camera_id,
        "frame_index": frame_index,
        "fires_detected": len(detected),
        "events": [inc.as_dict() for inc in detected],
    }


@app.get("/state", tags=["state"], response_class=JSONResponse)
async def get_state() -> dict[str, Any]:
    """
    Return the current warehouse state as JSON.

    Tries Redis first (shared across nodes); falls back to in-memory cache.
    """
    state: WarehouseState | None = None

    if _app_state.state_store:
        try:
            state = await _app_state.state_store.get_latest_state()
        except Exception as exc:
            logger.warning("Redis read error: %s", exc)

    if state is None:
        state = _app_state.get_latest_state()

    if state is None:
        raise HTTPException(status_code=503, detail="No state available yet.")

    return state.as_dict()


@app.get("/state/bev", tags=["state"])
async def get_bev_image() -> Response:
    """
    Return the current Bird's-Eye-View map as a JPEG image.

    The image is a top-down overhead render of the warehouse showing
    forklifts (orange), workers (green), AMRs (yellow), pallets (grey),
    and incident markers (red).
    """
    bev_bytes = _app_state._bev_cache

    if bev_bytes is None:
        # Generate a blank canvas
        bev_np = _app_state.estimator.get_bev_map()
        bev_bytes = _encode_jpeg(bev_np, settings.bev_jpeg_quality)

    return Response(content=bev_bytes, media_type="image/jpeg")


class IncidentFilter(BaseModel):
    """Query params for the /incidents endpoint."""

    limit: int = 50
    severity: str | None = None
    event_types: list[str] | None = None


@app.get("/incidents", tags=["safety"])
async def get_incidents(
    limit: int = 50,
    severity: str | None = None,
) -> dict[str, Any]:
    """
    Return recent active incidents.

    Queries the EventStore for persisted incidents; also merges any
    incidents from the current in-memory state.
    """
    state = _app_state.get_latest_state()
    active_in_memory: list[dict[str, Any]] = []

    if state:
        for inc in state.active_incidents:
            if severity and inc.severity != severity:
                continue
            active_in_memory.append(inc.as_dict())

    # Also query event store if available
    persisted: list[dict[str, Any]] = []
    if _app_state.event_store:
        import datetime

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        window = datetime.timedelta(hours=1)
        try:
            event_types = ["near_miss", "zone_violation", "collision", "fire"]
            events = await _app_state.event_store.query_events(
                start=now - window,
                end=now,
                event_types=event_types,
                limit=limit,
            )
            for ev in events:
                if severity and ev.severity != severity:
                    continue
                persisted.append(ev.as_dict())
        except Exception as exc:
            logger.warning("EventStore query error: %s", exc)

    # Merge and deduplicate
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in active_in_memory + persisted:
        key = item.get("incident_id") or item.get("event_id", "")
        if key not in seen:
            seen.add(key)
            merged.append(item)
            if len(merged) >= limit:
                break

    return {
        "count": len(merged),
        "incidents": merged,
        "warehouse_id": settings.warehouse_id,
    }


@app.get("/analytics/throughput", tags=["analytics"])
async def get_throughput(window_seconds: float | None = None) -> dict[str, Any]:
    """
    Return warehouse KPI analytics.

    Query params
    ------------
    window_seconds : float | None
        Override the default analytics window.  Default: 3600 s.
    """
    return {
        "warehouse_id": settings.warehouse_id,
        "kpis": _app_state.analytics.compute_all(window_seconds),
        "timestamp": time.time(),
    }


@app.get("/analytics/history", tags=["analytics"])
async def get_state_history(n: int = 50) -> dict[str, Any]:
    """Return the N most recent state snapshots from Redis."""
    if _app_state.state_store is None:
        raise HTTPException(status_code=503, detail="Redis not connected.")

    history = await _app_state.state_store.get_state_history(n=n)
    return {
        "count": len(history),
        "history": [s.as_dict() for s in history],
    }


# ---------------------------------------------------------------------------
# WebSocket — real-time stream
# ---------------------------------------------------------------------------


@app.post("/demo/seed", tags=["demo"])
async def demo_seed() -> dict[str, Any]:
    """
    Inject a realistic mock warehouse state so BEV/state/incidents have data.

    Creates 3 forklifts, 4 workers, 2 AMRs, 5 pallets, and 1 active incident.
    Safe to call repeatedly — each call refreshes the state.
    """
    import math

    forklifts = [
        AgentPose("FL-01", "forklift", x=15.0, y=30.0, z=0.0, heading_rad=0.0,    vx=1.2, vy=0.0, confidence=0.97),
        AgentPose("FL-02", "forklift", x=45.0, y=12.0, z=0.0, heading_rad=math.pi/4, vx=0.8, vy=0.8, confidence=0.95),
        AgentPose("FL-03", "forklift", x=75.0, y=48.0, z=0.0, heading_rad=math.pi,   vx=-0.5, vy=0.0, confidence=0.93),
    ]
    workers = [
        AgentPose("WK-01", "worker", x=20.0, y=20.0, z=0.0, heading_rad=1.2, confidence=0.91),
        AgentPose("WK-02", "worker", x=55.0, y=35.0, z=0.0, heading_rad=2.5, confidence=0.89),
        AgentPose("WK-03", "worker", x=30.0, y=50.0, z=0.0, heading_rad=0.3, confidence=0.94),
        AgentPose("WK-04", "worker", x=80.0, y=10.0, z=0.0, heading_rad=3.1, confidence=0.88),
    ]
    amrs = [
        AgentPose("AMR-01", "amr", x=60.0, y=25.0, z=0.0, heading_rad=1.57, vx=0.0, vy=1.5, confidence=0.99),
        AgentPose("AMR-02", "amr", x=25.0, y=42.0, z=0.0, heading_rad=4.71, vx=0.0, vy=-1.0, confidence=0.98),
    ]
    pallets = [
        InventoryLocation("PLT-001", barcode="BC001", x=10.0, y=5.0,  z=0.0, zone="A"),
        InventoryLocation("PLT-002", barcode="BC002", x=35.0, y=8.0,  z=0.0, zone="A"),
        InventoryLocation("PLT-003", barcode="BC003", x=62.0, y=5.0,  z=0.0, zone="B"),
        InventoryLocation("PLT-004", barcode="BC004", x=85.0, y=55.0, z=0.0, zone="B"),
        InventoryLocation("PLT-005", barcode="BC005", x=50.0, y=55.0, z=0.0, zone="A"),
    ]
    incidents = [
        Incident(
            incident_id="INC-001",
            incident_type="near_miss",
            severity="high",
            timestamp=time.monotonic(),
            description="FL-01 and WK-01 proximity alert — 1.2 m separation",
            agent_ids=["FL-01", "WK-01"],
            location_x=17.0,
            location_y=26.0,
        )
    ]

    state = WarehouseState(
        timestamp=time.monotonic(),
        frame_index=1,
        forklift_poses=forklifts,
        worker_poses=workers,
        amr_poses=amrs,
        inventory_locations=pallets,
        active_incidents=incidents,
        track_count=len(forklifts) + len(workers) + len(amrs),
    )

    # Render BEV and push into the estimator cache
    _app_state.estimator._active_incidents = incidents
    bev = _app_state.estimator._render_bev(forklifts, workers, amrs, pallets)
    _app_state.estimator._bev_cache = bev
    _app_state._bev_cache = _encode_jpeg(bev, settings.bev_jpeg_quality)

    _app_state.set_latest_state(state)

    return {
        "status": "seeded",
        "forklifts": len(forklifts),
        "workers": len(workers),
        "amrs": len(amrs),
        "pallets": len(pallets),
        "incidents": len(incidents),
        "view_bev": "http://localhost:8003/state/bev",
        "view_state": "http://localhost:8003/state",
    }


@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint: streams WarehouseState as JSON at ~10 Hz.

    Message format: JSON object (same as GET /state).

    Connection is rejected if the server is at capacity
    (DT_STREAM_MAX_CLIENTS).

    The client can send a JSON ping: {"type": "ping"} → {"type": "pong"}.
    """
    await websocket.accept()

    if not _app_state.add_ws_client(websocket):
        await websocket.send_json({"error": "Too many clients. Try again later."})
        await websocket.close(code=1013)  # Try Again Later
        return

    client_addr = websocket.client
    logger.info("WebSocket client connected: %s (total=%d)", client_addr, _app_state.ws_client_count)

    try:
        # Send current state immediately on connect
        state = _app_state.get_latest_state()
        if state:
            await websocket.send_text(json.dumps(state.as_dict()))

        while True:
            # Check for incoming messages (ping / control) with a short timeout
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong", "ts": time.time()}
                    )
            except asyncio.TimeoutError:
                pass
            except (json.JSONDecodeError, KeyError):
                pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", client_addr)
    except Exception as exc:
        logger.warning("WebSocket error (%s): %s", client_addr, exc)
    finally:
        _app_state.remove_ws_client(websocket)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR numpy array as JPEG bytes."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("Failed to encode BEV image as JPEG.")
    return bytes(buffer)


# ---------------------------------------------------------------------------
# Direct injection API (used by the ingestion pipeline in the same process)
# ---------------------------------------------------------------------------


def inject_state(state: WarehouseState) -> None:
    """
    Inject a new WarehouseState directly into the app state.

    Call this from the processing pipeline when the API and the estimator
    run in the same process (single-node deployment).
    """
    _app_state.sync.push_state(state)


def inject_incident(incident: Incident) -> None:
    """
    Inject an incident directly (e.g. from the safety_ai detectors).
    """
    state = _app_state.get_latest_state()
    if state:
        state.active_incidents.append(incident)


async def store_event(event: WarehouseEvent) -> None:
    """
    Persist a WarehouseEvent to the event store from within the same process.
    """
    if _app_state.event_store:
        await _app_state.event_store.write_event(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry point registered in pyproject.toml."""
    import uvicorn

    uvicorn.run(
        "warehousegpt.digital_twin.api.main:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

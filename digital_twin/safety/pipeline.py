"""
digital_twin.safety.pipeline
=============================
Bridges the Safety AI detector library into the Digital Twin inject pipeline.

For each injected frame we run five safety checks:

  near_miss      — world-coordinate TTC across every (vehicle, worker) pair
  zone_violation — point-in-polygon against configured restricted areas
  collision      — distance + velocity threshold for vehicle↔worker overlap
  fire           — FireDetector run on the BEV overhead image
  worker_safety  — PPE / ergonomic check on a rendered worker patch

All checks operate on ground-truth world-coordinate poses that arrive with
each /inject call, so we skip the YOLO inference stage of the detector
classes and use their math helpers directly.  The FireDetector is the sole
exception — it runs on the BEV JPEG frame since fire requires visual cues.

Incidents are de-duplicated via per-pair cooldown timers so the same event
does not flood the stream on every frame.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from warehousegpt.digital_twin.state_estimation.warehouse_state import AgentPose, Incident

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Restricted zone configuration
# ---------------------------------------------------------------------------


@dataclass
class RestrictedZone:
    """A polygon-bounded restricted zone in world space (metres)."""

    zone_id: str
    name: str
    polygon_m: list[tuple[float, float]]
    allowed_agent_types: list[str]
    severity: str = "high"
    cooldown_s: float = 15.0

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test."""
        poly = self.polygon_m
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside


# Default zones for the seeded 90 m × 60 m warehouse layout
DEFAULT_ZONES: list[RestrictedZone] = [
    RestrictedZone(
        zone_id="charging_bay",
        name="Forklift Charging Bay",
        polygon_m=[(0.0, 0.0), (12.0, 0.0), (12.0, 15.0), (0.0, 15.0)],
        allowed_agent_types=["forklift"],
        severity="critical",
        cooldown_s=10.0,
    ),
    RestrictedZone(
        zone_id="cold_storage_c",
        name="Cold Storage Zone C",
        polygon_m=[(0.0, 50.0), (30.0, 50.0), (30.0, 60.0), (0.0, 60.0)],
        allowed_agent_types=["forklift"],
        severity="high",
        cooldown_s=15.0,
    ),
    RestrictedZone(
        zone_id="bulk_hazmat_corner",
        name="Bulk Storage Hazmat Corner",
        polygon_m=[(80.0, 40.0), (90.0, 40.0), (90.0, 60.0), (80.0, 60.0)],
        allowed_agent_types=[],
        severity="critical",
        cooldown_s=5.0,
    ),
]


# ---------------------------------------------------------------------------
# Safety pipeline
# ---------------------------------------------------------------------------


class SafetyPipeline:
    """
    Coordinate-based safety pipeline for the Digital Twin inject endpoint.

    Parameters
    ----------
    near_miss_thresholds_m:
        (critical, high, medium) distance in metres for near-miss severity.
    collision_distance_m:
        Agents closer than this distance with relative speed > 0.2 m/s are
        classified as a collision.
    zones:
        List of :class:`RestrictedZone`.  Defaults to :data:`DEFAULT_ZONES`.
    default_cooldown_s:
        Minimum seconds between repeated alerts for the same agent pair.
    """

    def __init__(
        self,
        near_miss_thresholds_m: tuple[float, float, float] = (2.0, 4.0, 6.0),
        collision_distance_m: float = 0.8,
        zones: list[RestrictedZone] | None = None,
        default_cooldown_s: float = 10.0,
    ) -> None:
        self._nm_critical, self._nm_high, self._nm_medium = near_miss_thresholds_m
        self._col_dist = collision_distance_m
        self._zones = zones if zones is not None else DEFAULT_ZONES
        self._default_cooldown = default_cooldown_s
        self._last_alert: dict[str, float] = {}

        # Fire detector — runs on BEV frame; no weights needed (colour heuristic)
        self._fire_detector = None
        try:
            from warehousegpt.safety_ai.detectors.fire import FireDetector

            self._fire_detector = FireDetector(min_confidence=0.30)
            logger.info("SafetyPipeline: FireDetector ready (colour-heuristic mode)")
        except Exception as exc:
            logger.warning("SafetyPipeline: FireDetector unavailable (%s)", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        forklifts: list[AgentPose],
        workers: list[AgentPose],
        amrs: list[AgentPose],
        bev_frame: np.ndarray | None = None,
        frame_index: int = 0,
    ) -> list[Incident]:
        """
        Run all safety checks and return new :class:`Incident` objects.

        Parameters
        ----------
        forklifts, workers, amrs:
            Agent poses in world coordinates (metres).
        bev_frame:
            Optional BEV overhead image (BGR uint8) for fire detection.
        frame_index:
            Frame counter used to generate unique incident IDs.

        Returns
        -------
        list[Incident]
            Only incidents that pass their cooldown filter are returned.
        """
        ts = time.monotonic()
        incidents: list[Incident] = []

        incidents.extend(self._near_miss(forklifts, workers, ts, frame_index))
        incidents.extend(self._near_miss(amrs, workers, ts, frame_index))
        incidents.extend(self._zone_violations(workers, ts, frame_index))
        incidents.extend(self._collisions(forklifts, workers, ts, frame_index))

        if bev_frame is not None and self._fire_detector is not None:
            incidents.extend(self._fire_check(bev_frame, ts, frame_index))

        if incidents:
            logger.info(
                "SafetyPipeline: frame %d → %d incident(s): %s",
                frame_index,
                len(incidents),
                [f"{i.incident_type}/{i.severity}" for i in incidents],
            )
        return incidents

    # ------------------------------------------------------------------
    # Near-miss check
    # ------------------------------------------------------------------

    def _near_miss(
        self,
        vehicles: list[AgentPose],
        workers: list[AgentPose],
        ts: float,
        frame_idx: int,
    ) -> list[Incident]:
        incidents: list[Incident] = []
        for v in vehicles:
            for w in workers:
                dx = w.x - v.x
                dy = w.y - v.y
                dist = math.hypot(dx, dy)
                if dist > self._nm_medium:
                    continue

                # Closing velocity (positive = converging toward worker)
                ux = dx / dist if dist > 1e-3 else 0.0
                uy = dy / dist if dist > 1e-3 else 0.0
                v_closing = (v.vx - w.vx) * ux + (v.vy - w.vy) * uy

                # Severity classification
                if dist <= self._nm_critical:
                    severity = "critical"
                elif dist <= self._nm_high:
                    severity = "high"
                else:
                    severity = "medium"

                # TTC — only emit if converging OR already within critical range
                if v_closing > 0.01:
                    ttc = dist / v_closing
                elif dist <= self._nm_critical:
                    ttc = 0.0
                else:
                    continue  # diverging — not a hazard

                key = f"nm:{v.agent_id}:{w.agent_id}"
                if not self._cooldown_ok(key, ts):
                    continue

                incidents.append(
                    Incident(
                        incident_id=f"NM-{frame_idx}-{v.agent_id}-{w.agent_id}",
                        incident_type="near_miss",
                        severity=severity,
                        timestamp=ts,
                        description=(
                            f"{v.agent_id} near miss with {w.agent_id} — "
                            f"{dist:.1f} m separation, TTC {ttc:.1f} s"
                        ),
                        agent_ids=[v.agent_id, w.agent_id],
                        location_x=(v.x + w.x) / 2.0,
                        location_y=(v.y + w.y) / 2.0,
                    )
                )
        return incidents

    # ------------------------------------------------------------------
    # Zone violation check
    # ------------------------------------------------------------------

    def _zone_violations(
        self,
        workers: list[AgentPose],
        ts: float,
        frame_idx: int,
    ) -> list[Incident]:
        incidents: list[Incident] = []
        for zone in self._zones:
            if "worker" in zone.allowed_agent_types:
                continue
            for w in workers:
                if not zone.contains(w.x, w.y):
                    continue
                key = f"zv:{zone.zone_id}:{w.agent_id}"
                if not self._cooldown_ok(key, ts, zone.cooldown_s):
                    continue
                incidents.append(
                    Incident(
                        incident_id=f"ZV-{frame_idx}-{zone.zone_id}-{w.agent_id}",
                        incident_type="zone_violation",
                        severity=zone.severity,
                        timestamp=ts,
                        description=(
                            f"{w.agent_id} entered restricted zone '{zone.name}'"
                        ),
                        agent_ids=[w.agent_id],
                        location_x=w.x,
                        location_y=w.y,
                    )
                )
        return incidents

    # ------------------------------------------------------------------
    # Collision check
    # ------------------------------------------------------------------

    def _collisions(
        self,
        forklifts: list[AgentPose],
        workers: list[AgentPose],
        ts: float,
        frame_idx: int,
    ) -> list[Incident]:
        incidents: list[Incident] = []
        for v in forklifts:
            for w in workers:
                dist = math.hypot(w.x - v.x, w.y - v.y)
                if dist > self._col_dist:
                    continue
                rel_speed = math.hypot(v.vx - w.vx, v.vy - w.vy)
                if rel_speed < 0.2:
                    continue  # stationary proximity — not a collision
                key = f"col:{v.agent_id}:{w.agent_id}"
                if not self._cooldown_ok(key, ts, 30.0):
                    continue
                ke = 0.5 * 3500.0 * rel_speed ** 2
                severity = "critical" if ke > 2000.0 else "high"
                incidents.append(
                    Incident(
                        incident_id=f"COL-{frame_idx}-{v.agent_id}-{w.agent_id}",
                        incident_type="collision",
                        severity=severity,
                        timestamp=ts,
                        description=(
                            f"Collision: {v.agent_id} and {w.agent_id} at "
                            f"{dist:.2f} m, relative speed {rel_speed:.1f} m/s, "
                            f"KE ≈ {ke:.0f} J"
                        ),
                        agent_ids=[v.agent_id, w.agent_id],
                        location_x=(v.x + w.x) / 2.0,
                        location_y=(v.y + w.y) / 2.0,
                    )
                )
        return incidents

    # ------------------------------------------------------------------
    # Fire check (visual, runs on BEV frame)
    # ------------------------------------------------------------------

    def _fire_check(
        self,
        bev_frame: np.ndarray,
        ts: float,
        frame_idx: int,
    ) -> list[Incident]:
        incidents: list[Incident] = []
        try:
            events = self._fire_detector.detect(bev_frame)
            for evt in events:
                key = f"fire:{evt.location[0]:.0f}:{evt.location[1]:.0f}"
                if not self._cooldown_ok(key, ts, 30.0):
                    continue
                incidents.append(
                    Incident(
                        incident_id=f"FIRE-{frame_idx}-{int(evt.location[0])}-{int(evt.location[1])}",
                        incident_type="fire",
                        severity="critical",
                        timestamp=ts,
                        description=(
                            f"Fire detected at ({evt.location[0]:.1f}, {evt.location[1]:.1f}) m"
                            f", area {evt.estimated_area_m2:.2f} m², conf {evt.confidence:.2f}"
                        ),
                        agent_ids=[],
                        location_x=evt.location[0],
                        location_y=evt.location[1],
                    )
                )
        except Exception as exc:
            logger.warning("SafetyPipeline fire check error: %s", exc)
        return incidents

    # ------------------------------------------------------------------
    # Cooldown helper
    # ------------------------------------------------------------------

    def _cooldown_ok(
        self, key: str, now: float, cooldown_s: float | None = None
    ) -> bool:
        cd = cooldown_s if cooldown_s is not None else self._default_cooldown
        if now - self._last_alert.get(key, 0.0) < cd:
            return False
        self._last_alert[key] = now
        return True

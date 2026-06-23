"""
warehouse_state.py — WarehouseState dataclass and estimator.

WarehouseState
    Canonical snapshot of the warehouse at a given timestamp.

WarehouseStateEstimator
    Fuses multi-object tracker output with world-model predictions to
    produce WarehouseState instances at each camera frame.

BEV (Bird's Eye View) map
    A top-down occupancy map rendered as a numpy RGB image.
    Resolution: configurable via BEVConfig (default 2048×2048 px, 0.05 m/px).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from warehousegpt.digital_twin.sensor_fusion.multi_object_tracker import (
    Detection,
    MultiObjectTracker,
    Track,
    TrackState,
)


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AgentPose:
    """World-frame pose of a single tracked agent."""

    agent_id: str
    agent_type: str          # "forklift" | "worker" | "amr" | "unknown"
    x: float                 # world-frame metres
    y: float
    z: float
    heading_rad: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "heading_deg": round(float(np.degrees(self.heading_rad)), 2),
            "vx": round(self.vx, 4),
            "vy": round(self.vy, 4),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class InventoryLocation:
    """Real-time position of a pallet / inventory item."""

    item_id: str
    barcode: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    zone: str = "unknown"
    last_seen: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "barcode": self.barcode,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "zone": self.zone,
        }


@dataclass
class Incident:
    """Safety or operational incident."""

    incident_id: str
    incident_type: str       # "near_miss" | "zone_violation" | "collision" | …
    severity: str            # "low" | "medium" | "high" | "critical"
    timestamp: float
    description: str = ""
    agent_ids: list[str] = field(default_factory=list)
    location_x: float = 0.0
    location_y: float = 0.0
    resolved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "description": self.description,
            "agent_ids": self.agent_ids,
            "location": {"x": round(self.location_x, 4), "y": round(self.location_y, 4)},
            "resolved": self.resolved,
        }


# ---------------------------------------------------------------------------
# WarehouseState
# ---------------------------------------------------------------------------


@dataclass
class WarehouseState:
    """
    Full snapshot of the warehouse at a given point in time.

    Fields
    ------
    timestamp : float
        Monotonic wall-clock time (time.monotonic()).
    frame_index : int
        Sequential frame counter.
    forklift_poses : list[AgentPose]
        All tracked forklifts.
    worker_poses : list[AgentPose]
        All tracked workers (pedestrians).
    amr_poses : list[AgentPose]
        Autonomous mobile robots.
    inventory_locations : list[InventoryLocation]
        Tracked pallets / inventory items.
    occupancy_grid : np.ndarray
        2-D binary occupancy grid (0 = free, 1 = occupied).
        Shape: (grid_h, grid_w) uint8.
    active_incidents : list[Incident]
        Unresolved incidents at this timestamp.
    track_count : int
        Total number of live tracks (for monitoring).
    """

    timestamp: float = field(default_factory=time.monotonic)
    frame_index: int = 0
    forklift_poses: list[AgentPose] = field(default_factory=list)
    worker_poses: list[AgentPose] = field(default_factory=list)
    amr_poses: list[AgentPose] = field(default_factory=list)
    inventory_locations: list[InventoryLocation] = field(default_factory=list)
    occupancy_grid: np.ndarray = field(
        default_factory=lambda: np.zeros((200, 200), dtype=np.uint8)
    )
    active_incidents: list[Incident] = field(default_factory=list)
    track_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict (occupancy_grid omitted)."""
        return {
            "timestamp": round(self.timestamp, 6),
            "frame_index": self.frame_index,
            "track_count": self.track_count,
            "forklifts": [p.as_dict() for p in self.forklift_poses],
            "workers": [p.as_dict() for p in self.worker_poses],
            "amrs": [p.as_dict() for p in self.amr_poses],
            "inventory": [i.as_dict() for i in self.inventory_locations],
            "active_incidents": [inc.as_dict() for inc in self.active_incidents],
        }


# ---------------------------------------------------------------------------
# BEV configuration
# ---------------------------------------------------------------------------


@dataclass
class BEVConfig:
    """Configuration for the bird's-eye-view overhead map renderer."""

    # Warehouse dimensions in metres
    world_width_m: float = 100.0
    world_height_m: float = 60.0

    # Image resolution in pixels
    img_width_px: int = 2000
    img_height_px: int = 1200

    # Agent render radii (in metres, converted to px at render time)
    forklift_radius_m: float = 1.5
    worker_radius_m: float = 0.5
    amr_radius_m: float = 0.75
    pallet_radius_m: float = 0.6

    # Colours (BGR)
    bg_colour: tuple[int, int, int] = (30, 30, 30)
    forklift_colour: tuple[int, int, int] = (0, 180, 255)    # orange
    worker_colour: tuple[int, int, int] = (0, 255, 100)      # green
    amr_colour: tuple[int, int, int] = (255, 220, 0)         # cyan-yellow
    pallet_colour: tuple[int, int, int] = (160, 160, 160)    # grey
    incident_colour: tuple[int, int, int] = (0, 0, 255)      # red

    @property
    def scale_x(self) -> float:
        return self.img_width_px / self.world_width_m

    @property
    def scale_y(self) -> float:
        return self.img_height_px / self.world_height_m

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        """Convert world-frame (x, y) in metres to image pixel coordinates."""
        px = int(x * self.scale_x)
        py = int(self.img_height_px - y * self.scale_y)  # flip Y axis
        return px, py


# ---------------------------------------------------------------------------
# WarehouseStateEstimator
# ---------------------------------------------------------------------------


class WarehouseStateEstimator:
    """
    Fuses multi-object tracker output with optional world-model predictions
    to produce WarehouseState instances.

    Parameters
    ----------
    bev_config : BEVConfig | None
        Bird's-eye-view renderer configuration.
    tracker_dt : float
        Time step passed to the underlying MultiObjectTracker.
    world_bounds : tuple[float, float, float, float] | None
        (x_min, x_max, y_min, y_max) in metres — used to build occupancy grid.
    grid_resolution_m : float
        Metres per occupancy grid cell.
    incidents_ttl : float
        Time (seconds) after which an incident is auto-resolved.
    """

    def __init__(
        self,
        bev_config: BEVConfig | None = None,
        tracker_dt: float = 0.1,
        world_bounds: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 60.0),
        grid_resolution_m: float = 0.5,
        incidents_ttl: float = 30.0,
    ) -> None:
        self.bev_cfg = bev_config or BEVConfig()
        self._tracker = MultiObjectTracker(dt=tracker_dt)
        self._world_bounds = world_bounds           # (x_min, x_max, y_min, y_max)
        self._grid_res = grid_resolution_m
        self._incidents_ttl = incidents_ttl

        # Pre-compute grid shape
        x_min, x_max, y_min, y_max = world_bounds
        self._grid_w = int((x_max - x_min) / grid_resolution_m)
        self._grid_h = int((y_max - y_min) / grid_resolution_m)

        self._frame_index: int = 0
        self._active_incidents: list[Incident] = []
        self._inventory: dict[str, InventoryLocation] = {}

        # Latest rendered BEV image
        self._bev_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[Detection],
        dt: float | None = None,
        external_incidents: list[Incident] | None = None,
    ) -> WarehouseState:
        """
        Ingest a list of detections and produce an updated WarehouseState.

        Parameters
        ----------
        detections : list[Detection]
            Sensor-fused detections for this frame.
        dt : float | None
            Elapsed time since last frame; defaults to tracker.dt.
        external_incidents : list[Incident] | None
            Incidents detected by safety_ai this frame.

        Returns
        -------
        WarehouseState
            Updated warehouse snapshot.
        """
        self._frame_index += 1
        now = time.monotonic()

        # Run tracker
        tracks = self._tracker.track(detections, dt=dt)

        # Separate tracks by class
        forklift_poses: list[AgentPose] = []
        worker_poses: list[AgentPose] = []
        amr_poses: list[AgentPose] = []
        pallet_locs: list[InventoryLocation] = []

        for track in tracks:
            if track.state == TrackState.DEAD:
                continue
            pos = track.get_position()
            vel = track.get_velocity()
            pose = AgentPose(
                agent_id=track.track_id,
                agent_type=track.class_label,
                x=pos[0], y=pos[1], z=pos[2],
                heading_rad=track.get_heading(),
                vx=vel[0], vy=vel[1], vz=vel[2],
                confidence=track.confidence,
            )

            label = track.class_label.lower()
            if "forklift" in label:
                forklift_poses.append(pose)
            elif "worker" in label or "person" in label or "pedestrian" in label:
                worker_poses.append(pose)
            elif "amr" in label or "robot" in label:
                amr_poses.append(pose)
            elif "pallet" in label or "inventory" in label:
                loc = InventoryLocation(
                    item_id=track.track_id,
                    x=pos[0], y=pos[1], z=pos[2],
                    last_seen=now,
                )
                pallet_locs.append(loc)
                self._inventory[track.track_id] = loc

        # Expire old incidents
        self._active_incidents = [
            inc for inc in self._active_incidents
            if (now - inc.timestamp) < self._incidents_ttl and not inc.resolved
        ]

        # Merge external incidents
        if external_incidents:
            for inc in external_incidents:
                if not any(i.incident_id == inc.incident_id for i in self._active_incidents):
                    self._active_incidents.append(inc)

        # Build occupancy grid
        occ = self._build_occupancy_grid(forklift_poses, worker_poses, amr_poses)

        # Render BEV
        bev = self._render_bev(forklift_poses, worker_poses, amr_poses, pallet_locs)
        self._bev_cache = bev

        state = WarehouseState(
            timestamp=now,
            frame_index=self._frame_index,
            forklift_poses=forklift_poses,
            worker_poses=worker_poses,
            amr_poses=amr_poses,
            inventory_locations=list(self._inventory.values()),
            occupancy_grid=occ,
            active_incidents=list(self._active_incidents),
            track_count=len(tracks),
        )
        return state

    # ------------------------------------------------------------------
    # BEV rendering
    # ------------------------------------------------------------------

    def get_bev_map(self) -> np.ndarray:
        """Return the most recently rendered BEV image (BGR uint8)."""
        if self._bev_cache is None:
            # Return blank canvas
            cfg = self.bev_cfg
            img = np.full(
                (cfg.img_height_px, cfg.img_width_px, 3),
                cfg.bg_colour,
                dtype=np.uint8,
            )
            return img
        return self._bev_cache.copy()

    def _render_bev(
        self,
        forklifts: list[AgentPose],
        workers: list[AgentPose],
        amrs: list[AgentPose],
        pallets: list[InventoryLocation],
    ) -> np.ndarray:
        cfg = self.bev_cfg
        img = np.full(
            (cfg.img_height_px, cfg.img_width_px, 3),
            cfg.bg_colour,
            dtype=np.uint8,
        )

        def draw_agent(
            pose: AgentPose,
            radius_m: float,
            colour: tuple[int, int, int],
            draw_heading: bool = True,
        ) -> None:
            cx, cy = cfg.world_to_px(pose.x, pose.y)
            r_px = max(3, int(radius_m * cfg.scale_x))
            cv2.circle(img, (cx, cy), r_px, colour, -1)
            cv2.circle(img, (cx, cy), r_px, (255, 255, 255), 1)

            if draw_heading:
                # Draw a heading arrow
                arrow_len = r_px * 2
                dx = int(arrow_len * np.cos(pose.heading_rad))
                dy = -int(arrow_len * np.sin(pose.heading_rad))  # flip Y
                cv2.arrowedLine(
                    img, (cx, cy), (cx + dx, cy + dy),
                    (255, 255, 255), 1, tipLength=0.4,
                )

            # Label
            cv2.putText(
                img, pose.agent_id[:6],
                (cx + r_px + 2, cy),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (200, 200, 200), 1,
            )

        for f in forklifts:
            draw_agent(f, cfg.forklift_radius_m, cfg.forklift_colour)
        for w in workers:
            draw_agent(w, cfg.worker_radius_m, cfg.worker_colour, draw_heading=False)
        for a in amrs:
            draw_agent(a, cfg.amr_radius_m, cfg.amr_colour)

        for p in pallets:
            px, py = cfg.world_to_px(p.x, p.y)
            r = max(2, int(cfg.pallet_radius_m * cfg.scale_x))
            cv2.rectangle(img, (px - r, py - r), (px + r, py + r), cfg.pallet_colour, -1)

        # Draw active incidents as red circles
        for inc in self._active_incidents:
            if not inc.resolved:
                ix, iy = cfg.world_to_px(inc.location_x, inc.location_y)
                cv2.circle(img, (ix, iy), 20, cfg.incident_colour, 2)
                cv2.putText(
                    img, inc.incident_type[:10],
                    (ix + 22, iy),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, cfg.incident_colour, 1,
                )

        # Overlay grid lines (light grey, every 10 m)
        for x_m in np.arange(0, cfg.world_width_m, 10):
            px_x, _ = cfg.world_to_px(x_m, 0)
            cv2.line(img, (px_x, 0), (px_x, cfg.img_height_px), (50, 50, 50), 1)
        for y_m in np.arange(0, cfg.world_height_m, 10):
            _, py_y = cfg.world_to_px(0, y_m)
            cv2.line(img, (0, py_y), (cfg.img_width_px, py_y), (50, 50, 50), 1)

        return img

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def _build_occupancy_grid(
        self,
        forklifts: list[AgentPose],
        workers: list[AgentPose],
        amrs: list[AgentPose],
    ) -> np.ndarray:
        """2-D binary occupancy grid; 1 = occupied by a moving agent."""
        grid = np.zeros((self._grid_h, self._grid_w), dtype=np.uint8)
        x_min, x_max, y_min, y_max = self._world_bounds

        def mark(x: float, y: float, radius_m: float = 1.0) -> None:
            gx = int((x - x_min) / self._grid_res)
            gy = int((y - y_min) / self._grid_res)
            r = max(1, int(radius_m / self._grid_res))
            # Use OpenCV to fill a circle on the grid
            cv2.circle(grid, (gx, gy), r, 1, -1)

        for pose in forklifts:
            mark(pose.x, pose.y, 2.0)
        for pose in workers:
            mark(pose.x, pose.y, 0.6)
        for pose in amrs:
            mark(pose.x, pose.y, 1.0)

        return grid

    # ------------------------------------------------------------------
    # Incident management
    # ------------------------------------------------------------------

    def add_incident(self, incident: Incident) -> None:
        """Manually add an incident to the active list."""
        self._active_incidents.append(incident)

    def resolve_incident(self, incident_id: str) -> bool:
        """Mark an incident as resolved. Returns True if found."""
        for inc in self._active_incidents:
            if inc.incident_id == incident_id:
                inc.resolved = True
                return True
        return False

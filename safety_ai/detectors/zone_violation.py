"""
warehousegpt.safety_ai.detectors.zone_violation
================================================
Real-time restricted-zone violation detection.

Architecture
------------
1. Zone polygons are loaded from a YAML/JSON configuration file.
   Each zone carries:
       - id: str
       - name: str
       - polygon: list of (x_px, y_px) image-space vertices
       - allowed_classes: list[str]   (e.g. ["forklift"] → workers must stay out)
       - cooldown_s: float            (min seconds between repeated alerts per agent)
       - severity: str                ("warning" | "critical")

2. Every frame:
       a. YOLO detects agents (person, forklift, etc.).
       b. The bottom-centre point of each bounding box is used as the
          ground-contact point of the agent.
       c. ``cv2.pointPolygonTest`` checks containment for each (agent, zone) pair.
       d. If the agent class is not in the zone's ``allowed_classes`` list
          and the per-agent cooldown has expired, a ViolationEvent is emitted.

3. Cooldown is tracked per (agent_track_id, zone_id) to avoid alert storms.

Dependencies
------------
    pip install ultralytics opencv-python-headless numpy pyyaml
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ZoneConfig:
    """Configuration for a single restricted zone."""

    zone_id: str
    name: str
    polygon: np.ndarray  # shape (N, 2) float32
    allowed_classes: list[str]
    cooldown_s: float = 10.0
    severity: str = "warning"


@dataclass(slots=True)
class ViolationEvent:
    """A zone-crossing violation detected in a single frame."""

    zone_id: str
    zone_name: str
    agent_track_id: int
    agent_class: str
    severity: str

    foot_point: tuple[float, float]
    """Image-space (x, y) of the agent ground contact point."""

    bbox: tuple[int, int, int, int]
    """Bounding box (x1, y1, x2, y2) of the violating agent."""

    frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "agent_track_id": self.agent_track_id,
            "agent_class": self.agent_class,
            "severity": self.severity,
            "foot_point": list(self.foot_point),
            "bbox": list(self.bbox),
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Zone loader
# ---------------------------------------------------------------------------


def load_zones_from_config(config_path: str | Path) -> list[ZoneConfig]:
    """
    Load zone definitions from a YAML or JSON file.

    Expected YAML schema::

        zones:
          - id: "charging_bay"
            name: "Forklift Charging Bay"
            polygon: [[100, 200], [300, 200], [300, 400], [100, 400]]
            allowed_classes: ["forklift"]
            cooldown_s: 15.0
            severity: "critical"

    Parameters
    ----------
    config_path:
        Path to the .yaml or .json config file.

    Returns
    -------
    list[ZoneConfig]
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("Zone config not found: %s — using empty zone list.", path)
        return []

    raw: dict[str, Any] = {}
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]

            with path.open() as fh:
                raw = yaml.safe_load(fh)
        except ImportError:
            import json

            with path.open() as fh:
                raw = json.load(fh)
    else:
        import json

        with path.open() as fh:
            raw = json.load(fh)

    zones: list[ZoneConfig] = []
    for entry in raw.get("zones", []):
        polygon = np.array(entry["polygon"], dtype=np.float32)
        zones.append(
            ZoneConfig(
                zone_id=entry["id"],
                name=entry["name"],
                polygon=polygon,
                allowed_classes=entry.get("allowed_classes", []),
                cooldown_s=float(entry.get("cooldown_s", 10.0)),
                severity=entry.get("severity", "warning"),
            )
        )
    logger.info("Loaded %d restricted zones from %s", len(zones), path)
    return zones


def _default_zones(frame_shape: tuple[int, ...]) -> list[ZoneConfig]:
    """Provide two example zones when no config file is supplied."""
    h, w = frame_shape[:2]
    return [
        ZoneConfig(
            zone_id="zone_charging",
            name="Charging Bay",
            polygon=np.array(
                [[0, 0], [int(w * 0.2), 0], [int(w * 0.2), int(h * 0.3)], [0, int(h * 0.3)]],
                dtype=np.float32,
            ),
            allowed_classes=["forklift"],
            cooldown_s=10.0,
            severity="critical",
        ),
        ZoneConfig(
            zone_id="zone_hazmat",
            name="Hazmat Storage",
            polygon=np.array(
                [
                    [int(w * 0.7), int(h * 0.6)],
                    [w, int(h * 0.6)],
                    [w, h],
                    [int(w * 0.7), h],
                ],
                dtype=np.float32,
            ),
            allowed_classes=[],
            cooldown_s=5.0,
            severity="critical",
        ),
    ]


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

_CLASS_NAMES: dict[int, str] = {0: "person", 1: "forklift", 2: "rack"}


class ZoneViolationDetector:
    """
    Restricted-zone violation detector.

    Parameters
    ----------
    zones:
        List of :class:`ZoneConfig` objects, or path to YAML/JSON config file.
        Pass ``None`` to use built-in demo zones (frame-relative).
    yolo_model_path:
        Fine-tuned YOLOv8 model for detection.
    conf_threshold:
        Minimum detection confidence.

    Usage
    -----
    ::

        detector = ZoneViolationDetector(zones="config/zones.yaml")
        for frame in camera_feed:
            violations = detector.detect(frame)
            for v in violations:
                alert_system.send(v.to_dict())
    """

    def __init__(
        self,
        zones: list[ZoneConfig] | str | Path | None = None,
        yolo_model_path: str | None = None,
        conf_threshold: float = 0.45,
    ) -> None:
        self._conf = conf_threshold
        self._zones: list[ZoneConfig] = []
        self._zones_initialised = False

        if isinstance(zones, (str, Path)):
            self._zones = load_zones_from_config(zones)
            self._zones_initialised = True
        elif isinstance(zones, list):
            self._zones = zones
            self._zones_initialised = True

        # Cooldown tracking: (track_id, zone_id) -> last_alert_time
        self._cooldowns: dict[tuple[int, str], float] = {}

        # Simple ID counter for mock mode (no YOLO)
        self._next_id = 0
        self._prev_detections: list[tuple[int, str, tuple[int, int, int, int]]] = []

        self._yolo: object | None = None
        if yolo_model_path:
            self._load_yolo(yolo_model_path)

        self._frame_idx = 0

    def _load_yolo(self, path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._yolo = YOLO(path)
            logger.info("ZoneViolationDetector: YOLO loaded from %s", path)
        except ImportError:
            logger.warning("ultralytics not available — using mock detections.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[ViolationEvent]:
        """
        Process a single frame and return any zone violation events.

        Parameters
        ----------
        frame:
            HxWx3 uint8 BGR frame.

        Returns
        -------
        list[ViolationEvent]
        """
        self._frame_idx += 1

        if not self._zones_initialised:
            self._zones = _default_zones(frame.shape)
            self._zones_initialised = True

        detections = self._detect(frame)  # [(track_id, class_name, bbox_xyxy)]
        violations: list[ViolationEvent] = []

        for track_id, cls_name, bbox in detections:
            # Use bottom-centre of bbox as ground contact point
            x1, y1, x2, y2 = bbox
            foot = ((x1 + x2) / 2.0, float(y2))

            for zone in self._zones:
                if cls_name in zone.allowed_classes:
                    continue  # agent is allowed here

                if not self._point_in_polygon(foot, zone.polygon):
                    continue

                # Check cooldown
                key = (track_id, zone.zone_id)
                now = time.time()
                if now - self._cooldowns.get(key, 0.0) < zone.cooldown_s:
                    continue

                self._cooldowns[key] = now
                violations.append(
                    ViolationEvent(
                        zone_id=zone.zone_id,
                        zone_name=zone.name,
                        agent_track_id=track_id,
                        agent_class=cls_name,
                        severity=zone.severity,
                        foot_point=foot,
                        bbox=bbox,
                        frame_idx=self._frame_idx,
                    )
                )

        return violations

    def add_zone(self, zone: ZoneConfig) -> None:
        """Dynamically add or replace a zone at runtime."""
        self._zones = [z for z in self._zones if z.zone_id != zone.zone_id]
        self._zones.append(zone)
        self._zones_initialised = True
        logger.info("Added zone '%s' (%s)", zone.name, zone.zone_id)

    def remove_zone(self, zone_id: str) -> None:
        """Remove a zone by ID."""
        self._zones = [z for z in self._zones if z.zone_id != zone_id]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(
        self, frame: np.ndarray
    ) -> list[tuple[int, str, tuple[int, int, int, int]]]:
        if self._yolo is not None:
            return self._yolo_detect(frame)
        return self._mock_detect(frame)

    def _yolo_detect(
        self, frame: np.ndarray
    ) -> list[tuple[int, str, tuple[int, int, int, int]]]:
        results = self._yolo(frame, conf=self._conf, verbose=False)  # type: ignore[call-arg]
        out: list[tuple[int, str, tuple[int, int, int, int]]] = []
        for r in results:
            if r.boxes is None:
                continue
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                xyxy = tuple(box.xyxy[0].cpu().numpy().astype(int).tolist())
                # Use sequential index as proxy track ID (replace with real tracker)
                tid = self._next_id + i
                out.append((tid, cls_name, xyxy))  # type: ignore[arg-type]
            self._next_id += len(r.boxes) if r.boxes else 0
        return out

    def _mock_detect(
        self, frame: np.ndarray
    ) -> list[tuple[int, str, tuple[int, int, int, int]]]:
        h, w = frame.shape[:2]
        return [
            (
                0,
                "person",
                (int(0.05 * w), int(0.10 * h), int(0.12 * w), int(0.28 * h)),
            ),
            (
                1,
                "forklift",
                (int(0.75 * w), int(0.65 * h), int(0.95 * w), int(0.95 * h)),
            ),
        ]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _point_in_polygon(
        point: tuple[float, float], polygon: np.ndarray
    ) -> bool:
        """
        Test if (x, y) point lies inside the given polygon using
        cv2.pointPolygonTest when available, or a pure-NumPy ray-casting
        fallback.
        """
        try:
            import cv2  # type: ignore[import-untyped]

            result = cv2.pointPolygonTest(
                polygon.reshape(-1, 1, 2), point, measureDist=False
            )
            return result >= 0
        except ImportError:
            pass

        # Pure-NumPy ray-casting
        x, y = point
        poly = polygon
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
            ):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers (useful for testing)."""
        self._cooldowns.clear()

    def zone_summary(self) -> list[dict[str, object]]:
        """Return current zone configurations as a list of dicts."""
        return [
            {
                "zone_id": z.zone_id,
                "name": z.name,
                "vertices": z.polygon.tolist(),
                "allowed_classes": z.allowed_classes,
                "cooldown_s": z.cooldown_s,
                "severity": z.severity,
            }
            for z in self._zones
        ]

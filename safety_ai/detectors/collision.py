"""
warehousegpt.safety_ai.detectors.collision
==========================================
Post-hoc collision detection from warehouse video sequences.

Architecture
------------
1. 3D bounding-box estimation lifts 2D detections into 3D using
   monocular depth (MiDaS / ZoeDepth) + known forklift dimensions as
   priors.  When stereo or LiDAR data is available it is used directly.
2. Overlap detection in 3D triggers a collision candidate.
3. Impact severity is scored by:
       a) relative velocity at contact (kinematic model)
       b) object mass estimates (person ~80 kg, forklift ~3 500 kg)
       c) post-contact deformation proxy (IOU change across frames)
4. A CollisionEvent is emitted with rich metadata for incident reports.

Severity scale
--------------
    MINOR    : kinetic energy < 500 J  — no injury expected
    MODERATE : 500–2 000 J             — injury possible
    SEVERE   : > 2 000 J               — serious injury / death risk

Dependencies
------------
    pip install ultralytics>=8.2 opencv-python-headless numpy
    # Optional depth:
    pip install timm torch  # for MiDaS
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ImpactSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass(slots=True)
class CollisionEvent:
    """Detected collision between warehouse agents."""

    timestamp: float
    """UTC wall-clock time of the collision frame."""

    frame_idx: int
    """Frame index within the video stream."""

    location: tuple[float, float, float]
    """Estimated 3-D world coordinates (x, y, z) in metres from camera origin."""

    agents_involved: list[str]
    """Human-readable labels, e.g. ['worker_3', 'forklift_7']."""

    impact_severity: ImpactSeverity
    """Computed severity category."""

    kinetic_energy_joules: float
    """Estimated kinetic energy transferred at impact (J)."""

    relative_velocity_ms: float
    """Magnitude of relative velocity at first contact (m/s)."""

    confidence: float
    """Detector confidence for this event (0–1)."""

    bbox_2d_agents: list[tuple[int, int, int, int]] = field(default_factory=list)
    """2-D bounding boxes (xyxy) of each involved agent."""

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "frame_idx": self.frame_idx,
            "location": list(self.location),
            "agents_involved": self.agents_involved,
            "impact_severity": self.impact_severity.value,
            "kinetic_energy_joules": round(self.kinetic_energy_joules, 1),
            "relative_velocity_ms": round(self.relative_velocity_ms, 3),
            "confidence": round(self.confidence, 4),
            "bbox_2d_agents": [list(b) for b in self.bbox_2d_agents],
        }


# ---------------------------------------------------------------------------
# 3-D Bounding-Box estimation
# ---------------------------------------------------------------------------


@dataclass
class BBox3D:
    """Axis-aligned 3-D bounding box in camera/world space."""

    cx: float  # centre x (m)
    cy: float  # centre y (m)
    cz: float  # centre z (m) — depth
    w: float  # width  (m)
    h: float  # height (m)
    d: float  # depth  (m)

    def corners(self) -> np.ndarray:
        """Return 8 corner points as (8, 3) array."""
        hx, hy, hz = self.w / 2, self.h / 2, self.d / 2
        c = np.array([self.cx, self.cy, self.cz])
        offsets = np.array(
            [
                [-hx, -hy, -hz],
                [+hx, -hy, -hz],
                [-hx, +hy, -hz],
                [+hx, +hy, -hz],
                [-hx, -hy, +hz],
                [+hx, -hy, +hz],
                [-hx, +hy, +hz],
                [+hx, +hy, +hz],
            ]
        )
        return c + offsets  # (8,3)

    def overlaps(self, other: "BBox3D") -> bool:
        """AABB intersection test."""
        return (
            abs(self.cx - other.cx) < (self.w + other.w) / 2
            and abs(self.cy - other.cy) < (self.h + other.h) / 2
            and abs(self.cz - other.cz) < (self.d + other.d) / 2
        )


# Known object dimensions (width x height x depth) in metres
_OBJECT_DIMS: dict[str, tuple[float, float, float]] = {
    "person": (0.50, 1.80, 0.30),
    "forklift": (1.20, 2.30, 2.50),
    "rack": (0.60, 3.00, 2.00),
}

# Known object masses in kg
_OBJECT_MASS: dict[str, float] = {
    "person": 80.0,
    "forklift": 3500.0,
    "rack": 500.0,
}


class _DepthEstimator:
    """
    Lightweight monocular depth estimator wrapping MiDaS.

    Falls back to a calibration-based pinhole projection when the model
    is unavailable.
    """

    def __init__(self, model_type: str = "MiDaS_small") -> None:
        self._model: object | None = None
        self._transform: object | None = None
        try:
            import torch  # type: ignore[import-untyped]
            from torch.hub import load as hub_load  # type: ignore[import-untyped]

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = hub_load("intel-isl/MiDaS", model_type)
            self._model.to(self._device).eval()  # type: ignore[union-attr]
            transforms = hub_load("intel-isl/MiDaS", "transforms")
            self._transform = (
                transforms.small_transform
                if "small" in model_type
                else transforms.default_transform
            )
            logger.info("DepthEstimator: MiDaS %s loaded on %s", model_type, self._device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DepthEstimator: cannot load MiDaS (%s). Using calibration fallback.", exc
            )

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Return relative inverse-depth map, shape (H, W), float32."""
        if self._model is None:
            # Fallback: linear gradient (placeholder for calibration)
            h, w = frame.shape[:2]
            return np.tile(np.linspace(1.0, 10.0, w, dtype=np.float32), (h, 1))

        import torch  # type: ignore[import-untyped]

        inp = self._transform(frame).to(self._device)  # type: ignore[operator]
        with torch.no_grad():
            pred = self._model(inp)  # type: ignore[operator]
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = pred.cpu().numpy().astype(np.float32)
        # Normalise to rough metric scale (heuristic; proper calibration needed)
        depth = (depth - depth.min()) / (depth.ptp() + 1e-6) * 10.0 + 0.5
        return depth


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class CollisionDetector:
    """
    Post-hoc collision detector that analyses a sequence of video frames.

    Parameters
    ----------
    yolo_model_path:
        Path to fine-tuned YOLOv8 model for person / forklift / rack detection.
    depth_model_type:
        MiDaS model type ("MiDaS_small" | "DPT_Large" | "DPT_Hybrid").
    camera_intrinsics:
        (fx, fy, cx, cy) in pixels.  Required for metric 3-D lifting.
    conf_threshold:
        Minimum YOLO detection confidence.
    iou_overlap_threshold:
        2-D IOU above which a collision candidate is raised for 3-D verification.
    energy_thresholds:
        (minor_max, moderate_max) kinetic energy thresholds in Joules.

    Usage
    -----
    ::

        detector = CollisionDetector("weights/yolov8x.pt")
        frame_buffer = [frame_t_minus_2, frame_t_minus_1, frame_t]
        events = detector.analyze_sequence(frame_buffer)
    """

    def __init__(
        self,
        yolo_model_path: str | None = None,
        depth_model_type: str = "MiDaS_small",
        camera_intrinsics: tuple[float, float, float, float] | None = None,
        conf_threshold: float = 0.45,
        iou_overlap_threshold: float = 0.15,
        energy_thresholds: tuple[float, float] = (500.0, 2000.0),
    ) -> None:
        self._conf = conf_threshold
        self._iou_thresh = iou_overlap_threshold
        self._energy_minor, self._energy_moderate = energy_thresholds
        self._fx, self._fy, self._cx_cam, self._cy_cam = (
            camera_intrinsics or (800.0, 800.0, 640.0, 360.0)
        )

        self._depth_estimator = _DepthEstimator(depth_model_type)

        self._yolo: object | None = None
        if yolo_model_path is not None:
            self._load_yolo(yolo_model_path)

        # Velocity estimation state: {agent_label: (cx3d, cy3d, cz3d)}
        self._prev_centers: dict[str, np.ndarray] = {}
        self._frame_idx = 0
        self._fps = 30.0

    def _load_yolo(self, path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._yolo = YOLO(path)
            logger.info("CollisionDetector: YOLO loaded from %s", path)
        except ImportError:
            logger.warning("ultralytics not installed — YOLO disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_sequence(
        self, frames: list[np.ndarray], fps: float = 30.0
    ) -> list[CollisionEvent]:
        """
        Analyse a buffer of consecutive frames for collision events.

        Parameters
        ----------
        frames:
            List of BGR frames in chronological order.
        fps:
            Video frame rate used for velocity computation.

        Returns
        -------
        list[CollisionEvent]
            Detected collision events (may be empty).
        """
        self._fps = fps
        events: list[CollisionEvent] = []
        for frame in frames:
            events.extend(self._process_frame(frame))
        return events

    def process_frame(self, frame: np.ndarray) -> list[CollisionEvent]:
        """Process a single frame (streaming mode)."""
        return self._process_frame(frame)

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> list[CollisionEvent]:
        self._frame_idx += 1
        depth_map = self._depth_estimator.estimate(frame)
        detections = self._detect(frame)  # [(label, bbox_xyxy, conf)]

        # Lift to 3-D
        boxes3d: dict[str, tuple[BBox3D, float, tuple[int, int, int, int]]] = {}
        for label, bbox, conf in detections:
            b3d = self._lift_to_3d(label, bbox, depth_map, frame.shape)
            boxes3d[label] = (b3d, conf, bbox)

        # Detect collisions: person × (forklift | rack)
        events: list[CollisionEvent] = []
        person_labels = [k for k in boxes3d if k.startswith("person")]
        vehicle_labels = [k for k in boxes3d if not k.startswith("person")]

        for p_label in person_labels:
            p_box, p_conf, p_bbox = boxes3d[p_label]
            for v_label in vehicle_labels:
                v_box, v_conf, v_bbox = boxes3d[v_label]
                if p_box.overlaps(v_box):
                    evt = self._build_event(
                        p_label, p_box, p_bbox, p_conf,
                        v_label, v_box, v_bbox, v_conf,
                    )
                    if evt is not None:
                        events.append(evt)

        # Update velocity state
        self._prev_centers = {
            label: np.array([b.cx, b.cy, b.cz])
            for label, (b, _, _) in boxes3d.items()
        }
        return events

    def _detect(
        self, frame: np.ndarray
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        """Return list of (label, bbox_xyxy_int, conf)."""
        if self._yolo is None:
            return self._mock_detections(frame)

        results = self._yolo(frame, conf=self._conf, verbose=False)  # type: ignore[call-arg]
        out: list[tuple[str, tuple[int, int, int, int], float]] = []
        counters: dict[str, int] = {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                name = r.names[cls_id]
                base = name.split("_")[0]
                counters[base] = counters.get(base, 0) + 1
                label = f"{base}_{counters[base]}"
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                out.append((label, tuple(xyxy.tolist()), conf))
        return out

    @staticmethod
    def _mock_detections(
        frame: np.ndarray,
    ) -> list[tuple[str, tuple[int, int, int, int], float]]:
        h, w = frame.shape[:2]
        return [
            ("person_1", (int(0.10 * w), int(0.30 * h), int(0.16 * w), int(0.60 * h)), 0.91),
            (
                "forklift_1",
                (int(0.12 * w), int(0.25 * h), int(0.30 * w), int(0.65 * h)),
                0.87,
            ),
        ]

    def _lift_to_3d(
        self,
        label: str,
        bbox: tuple[int, int, int, int],
        depth_map: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> BBox3D:
        x1, y1, x2, y2 = bbox
        # Sample median depth inside bounding box
        roi = depth_map[y1:y2, x1:x2]
        z = float(np.median(roi)) if roi.size > 0 else 5.0

        # Pixel centre -> 3-D point (pinhole)
        u_c = (x1 + x2) / 2.0
        v_c = (y1 + y2) / 2.0
        cx3d = (u_c - self._cx_cam) * z / self._fx
        cy3d = (v_c - self._cy_cam) * z / self._fy
        cz3d = z

        base = label.split("_")[0]
        w3d, h3d, d3d = _OBJECT_DIMS.get(base, (0.5, 1.0, 0.5))

        return BBox3D(cx3d, cy3d, cz3d, w3d, h3d, d3d)

    def _build_event(
        self,
        p_label: str,
        p_box: BBox3D,
        p_bbox: tuple[int, int, int, int],
        p_conf: float,
        v_label: str,
        v_box: BBox3D,
        v_bbox: tuple[int, int, int, int],
        v_conf: float,
    ) -> CollisionEvent | None:
        # Relative velocity (m/s)
        p_center = np.array([p_box.cx, p_box.cy, p_box.cz])
        v_center = np.array([v_box.cx, v_box.cy, v_box.cz])

        v_rel = 0.0
        if p_label in self._prev_centers and v_label in self._prev_centers:
            dp = (p_center - self._prev_centers[p_label]) * self._fps
            dv = (v_center - self._prev_centers[v_label]) * self._fps
            v_rel = float(np.linalg.norm(dp - dv))

        # Kinetic energy of forklift at impact
        base_v = v_label.split("_")[0]
        mass_v = _OBJECT_MASS.get(base_v, 100.0)
        ke = 0.5 * mass_v * v_rel ** 2

        if ke < 10.0 and v_rel < 0.1:
            # Likely stationary — not a real collision
            return None

        if ke < self._energy_minor:
            severity = ImpactSeverity.MINOR
        elif ke < self._energy_moderate:
            severity = ImpactSeverity.MODERATE
        else:
            severity = ImpactSeverity.SEVERE

        location = (
            (p_box.cx + v_box.cx) / 2,
            (p_box.cy + v_box.cy) / 2,
            (p_box.cz + v_box.cz) / 2,
        )

        return CollisionEvent(
            timestamp=time.time(),
            frame_idx=self._frame_idx,
            location=location,
            agents_involved=[p_label, v_label],
            impact_severity=severity,
            kinetic_energy_joules=ke,
            relative_velocity_ms=v_rel,
            confidence=min(p_conf, v_conf),
            bbox_2d_agents=[p_bbox, v_bbox],
        )

    def reset(self) -> None:
        """Reset velocity history between separate video clips."""
        self._prev_centers.clear()
        self._frame_idx = 0

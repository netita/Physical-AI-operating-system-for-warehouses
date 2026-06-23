"""
warehousegpt.safety_ai.detectors.near_miss
==========================================
Near-miss detection between workers and forklifts.

Architecture
------------
1. YOLOv8-X backbone (fine-tuned on warehouse data) detects and tracks
   persons and forklifts in every frame.
2. A Kalman-filter tracker maintains per-object state:
       [x, y, vx, vy, w, h]
   across frames to produce smooth velocity estimates.
3. Time-to-Collision (TTC) is computed from the relative velocity of
   each (person, forklift) pair and their current separation:
       TTC = d_gap / max(v_closing, epsilon)
   where d_gap is the gap between bounding-box edges projected onto the
   closing direction.
4. Events whose TTC falls below configurable thresholds are emitted as
   NearMissEvent with a severity tag.

Severity thresholds (configurable):
    CRITICAL  : TTC <= 1.5 s
    HIGH      : TTC <= 3.0 s
    MEDIUM    : TTC <= 5.0 s

Dependencies
------------
    pip install ultralytics>=8.2  opencv-python-headless>=4.10  numpy>=1.26
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
# Public data structures
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class NearMissEvent:
    """A single near-miss event emitted by :class:`NearMissDetector`."""

    severity: Severity
    """Qualitative severity derived from TTC."""

    involved_agents: list[int]
    """Tracker IDs of the agents involved (person_id, forklift_id)."""

    bbox: tuple[int, int, int, int]
    """Merged bounding box (x1, y1, x2, y2) covering both agents."""

    ttc_seconds: float
    """Estimated time-to-collision in seconds."""

    frame_idx: int = 0
    """Frame index within the current stream."""

    timestamp: float = field(default_factory=time.time)
    """Wall-clock UTC timestamp of detection."""

    confidence: float = 0.0
    """Detection confidence (0–1) of the lower-confidence agent."""

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity.value,
            "involved_agents": self.involved_agents,
            "bbox": list(self.bbox),
            "ttc_seconds": round(self.ttc_seconds, 3),
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "confidence": round(self.confidence, 4),
        }


# ---------------------------------------------------------------------------
# Internal Kalman tracker
# ---------------------------------------------------------------------------

_PERSON_CLASS_ID = 0
_FORKLIFT_CLASS_ID = 1


class _KalmanTrack:
    """Minimal constant-velocity Kalman filter for one detected object."""

    _F = np.array(
        [
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    _H = np.eye(6, dtype=np.float32)
    _Q = np.eye(6, dtype=np.float32) * 1e-2
    _R = np.eye(6, dtype=np.float32) * 1e-1

    def __init__(self, track_id: int, cls: int, bbox: np.ndarray, conf: float) -> None:
        self.track_id = track_id
        self.cls = cls
        self.conf = conf
        self.hits = 1
        self.misses = 0
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32)

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q

    def update(self, bbox: np.ndarray, conf: float) -> None:
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        z = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float32)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        self.conf = conf
        self.hits += 1
        self.misses = 0

    @property
    def velocity_px(self) -> np.ndarray:
        """Returns [vx, vy] in pixels/frame."""
        return self.x[2:4]

    @property
    def center(self) -> np.ndarray:
        """Returns [cx, cy] in pixels."""
        return self.x[:2]

    @property
    def bbox_xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[0], self.x[1], self.x[4], self.x[5]
        return np.array(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            dtype=np.float32,
        )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class NearMissDetector:
    """
    YOLOv8-based near-miss detector for persons and forklifts.

    Parameters
    ----------
    model_path:
        Path to a fine-tuned YOLOv8 .pt or .engine model file.
        When not provided the class can still be used in mock / test mode.
    conf_threshold:
        Minimum detection confidence to consider (default 0.4).
    ttc_thresholds:
        Mapping from :class:`Severity` to TTC upper bound (seconds).
    max_missed_frames:
        How many consecutive missed detections before a track is dropped.
    pixels_per_meter:
        Calibration constant used to convert pixel-velocity to m/s.
    fps:
        Expected frames-per-second of the input stream.

    Usage
    -----
    ::

        detector = NearMissDetector(model_path="weights/yolov8x_warehouse.pt", fps=30)
        for frame in video_stream:
            events = detector.predict(frame)
            for evt in events:
                logger.warning("Near-miss! %s", evt.to_dict())
    """

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.40,
        ttc_thresholds: dict[Severity, float] | None = None,
        max_missed_frames: int = 5,
        pixels_per_meter: float = 40.0,
        fps: float = 30.0,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._max_missed = max_missed_frames
        self._ppm = pixels_per_meter  # pixels / metre
        self._fps = fps
        self._ttc_thresholds: dict[Severity, float] = ttc_thresholds or {
            Severity.CRITICAL: 1.5,
            Severity.HIGH: 3.0,
            Severity.MEDIUM: 5.0,
        }

        self._tracks: dict[int, _KalmanTrack] = {}
        self._next_id: int = 0
        self._frame_idx: int = 0

        self._model: object | None = None
        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(model_path)
            logger.info("NearMissDetector: loaded model from %s", model_path)
        except ImportError:
            logger.warning(
                "ultralytics not installed — NearMissDetector running in mock mode."
            )

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def predict(self, frame: np.ndarray) -> list[NearMissEvent]:
        """
        Run near-miss detection on a single BGR frame.

        Parameters
        ----------
        frame:
            HxWx3 uint8 NumPy array in BGR order (OpenCV convention).

        Returns
        -------
        list[NearMissEvent]
            Zero or more events sorted by ascending TTC (most urgent first).
        """
        self._frame_idx += 1

        # --- 1. Detect objects ------------------------------------------------
        raw_detections = self._detect(frame)  # list of (cls, bbox_xyxy, conf)

        # --- 2. Update Kalman tracks ------------------------------------------
        self._update_tracks(raw_detections)

        # --- 3. Compute TTC for every (person, forklift) pair ----------------
        events = self._compute_near_misses()

        events.sort(key=lambda e: e.ttc_seconds)
        return events

    def _detect(
        self, frame: np.ndarray
    ) -> list[tuple[int, np.ndarray, float]]:
        """Return raw detections as list of (class_id, bbox_xyxy, confidence)."""
        if self._model is None:
            return self._mock_detections(frame)

        results = self._model(frame, conf=self._conf_threshold, verbose=False)  # type: ignore[call-arg]
        detections: list[tuple[int, np.ndarray, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in (_PERSON_CLASS_ID, _FORKLIFT_CLASS_ID):
                    continue
                conf = float(box.conf[0].item())
                if conf < self._conf_threshold:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append((cls_id, xyxy, conf))
        return detections

    @staticmethod
    def _mock_detections(
        frame: np.ndarray,
    ) -> list[tuple[int, np.ndarray, float]]:
        """Return synthetic detections for unit testing without GPU."""
        h, w = frame.shape[:2]
        return [
            (_PERSON_CLASS_ID, np.array([0.1 * w, 0.3 * h, 0.15 * w, 0.6 * h]), 0.92),
            (
                _FORKLIFT_CLASS_ID,
                np.array([0.5 * w, 0.2 * h, 0.65 * w, 0.7 * h]),
                0.88,
            ),
        ]

    def _update_tracks(
        self, detections: list[tuple[int, np.ndarray, float]]
    ) -> None:
        """Hungarian-assignment-free greedy IoU tracker (sufficient at ~30 fps)."""
        for track in self._tracks.values():
            track.predict()

        matched_track_ids: set[int] = set()

        for cls_id, bbox, conf in detections:
            best_iou, best_tid = 0.0, -1
            for tid, track in self._tracks.items():
                if track.cls != cls_id:
                    continue
                iou = _iou(bbox, track.bbox_xyxy)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou > 0.30 and best_tid != -1:
                self._tracks[best_tid].update(bbox, conf)
                matched_track_ids.add(best_tid)
            else:
                # New track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _KalmanTrack(tid, cls_id, bbox, conf)
                matched_track_ids.add(tid)

        # Increment miss counters and prune stale tracks
        stale = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.misses += 1
                if track.misses > self._max_missed:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

    def _compute_near_misses(self) -> list[NearMissEvent]:
        persons = {
            tid: t for tid, t in self._tracks.items() if t.cls == _PERSON_CLASS_ID
        }
        forklifts = {
            tid: t for tid, t in self._tracks.items() if t.cls == _FORKLIFT_CLASS_ID
        }

        events: list[NearMissEvent] = []

        for pid, person in persons.items():
            for fid, forklift in forklifts.items():
                ttc = self._ttc(person, forklift)
                severity = self._classify_ttc(ttc)
                if severity is None:
                    continue

                # Merged bounding box
                pb, fb = person.bbox_xyxy, forklift.bbox_xyxy
                merged_bbox = (
                    int(min(pb[0], fb[0])),
                    int(min(pb[1], fb[1])),
                    int(max(pb[2], fb[2])),
                    int(max(pb[3], fb[3])),
                )

                events.append(
                    NearMissEvent(
                        severity=severity,
                        involved_agents=[pid, fid],
                        bbox=merged_bbox,
                        ttc_seconds=ttc,
                        frame_idx=self._frame_idx,
                        confidence=min(person.conf, forklift.conf),
                    )
                )
        return events

    def _ttc(self, person: _KalmanTrack, forklift: _KalmanTrack) -> float:
        """
        Estimate TTC (seconds) between person and forklift.

        Uses closing speed along the axis connecting their centroids.
        TTC = gap_distance / closing_speed
        """
        rel_pos = person.center - forklift.center  # vector from forklift -> person
        dist = float(np.linalg.norm(rel_pos))
        if dist < 1e-3:
            return 0.0

        # Edge-to-edge gap (pixels)
        pb, fb = person.bbox_xyxy, forklift.bbox_xyxy
        gap_x = max(0.0, max(fb[0] - pb[2], pb[0] - fb[2]))
        gap_y = max(0.0, max(fb[1] - pb[3], pb[1] - fb[3]))
        gap_px = float(np.hypot(gap_x, gap_y))

        # Closing velocity: project relative velocity onto closing axis
        direction = rel_pos / dist
        # Relative velocity of person w.r.t. forklift (pixels/frame)
        v_rel_px = person.velocity_px - forklift.velocity_px
        # Positive closing speed means moving toward each other
        v_closing_px = -float(np.dot(v_rel_px, direction))  # forklift toward person
        v_closing_ms = v_closing_px / self._ppm * self._fps  # m/s

        if v_closing_ms <= 0.01:
            return float("inf")  # Diverging

        gap_m = gap_px / self._ppm
        return gap_m / v_closing_ms

    def _classify_ttc(self, ttc: float) -> Severity | None:
        thresholds = [
            (Severity.CRITICAL, self._ttc_thresholds[Severity.CRITICAL]),
            (Severity.HIGH, self._ttc_thresholds[Severity.HIGH]),
            (Severity.MEDIUM, self._ttc_thresholds[Severity.MEDIUM]),
        ]
        for sev, threshold in thresholds:
            if ttc <= threshold:
                return sev
        return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracks and reset frame counter (call between scenes)."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_idx = 0

    def active_tracks(self) -> list[dict[str, object]]:
        """Return current track state for diagnostics / visualisation."""
        return [
            {
                "track_id": t.track_id,
                "cls": "person" if t.cls == _PERSON_CLASS_ID else "forklift",
                "center": t.center.tolist(),
                "velocity_px": t.velocity_px.tolist(),
                "conf": round(t.conf, 4),
                "hits": t.hits,
            }
            for t in self._tracks.values()
        ]

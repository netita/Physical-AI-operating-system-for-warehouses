"""
multi_object_tracker.py — Hungarian-algorithm multi-object tracker.

Manages a pool of WarehouseKalmanFilter instances, one per tracked agent.
Implements track birth, track coasting (missing detections), and track
death logic.  Association uses the Hungarian algorithm via
scipy.optimize.linear_sum_assignment on a Mahalanobis distance cost matrix.

Supported agent classes: "forklift", "worker", "pallet", "obstacle"
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from warehousegpt.digital_twin.sensor_fusion.kalman_filter import WarehouseKalmanFilter

# Maximum cost for a valid assignment
_MAX_COST = 9999.0
# Mahalanobis distance gate — associations above this are rejected
_GATE_DISTANCE = 5.0


class TrackState(Enum):
    TENTATIVE = auto()   # newly born; not yet confirmed
    CONFIRMED = auto()   # matched enough times
    COASTING  = auto()   # missed for 1+ steps; still alive
    DEAD      = auto()   # will be removed from the pool


@dataclass
class Detection:
    """
    A single detection from one or more sensors, ready to be associated.

    Fields
    ------
    position : (3,) float64
        (x, y, z) in world frame.
    velocity : (3,) float64 | None
        (vx, vy, vz) if available (e.g. from Doppler LiDAR or optical flow).
    heading : float | None
        Yaw in radians.
    class_label : str
        Agent class: "forklift", "worker", "pallet", …
    confidence : float
        Detector confidence [0, 1].
    bbox_2d : (4,) float64 | None
        (cx, cy, w, h) pixel bounding box if from camera.
    sensor : str
        Source sensor: "camera", "lidar", "fusion".
    """

    position: np.ndarray                # (3,) float64
    velocity: Optional[np.ndarray] = None   # (3,) float64 | None
    heading: Optional[float] = None
    class_label: str = "unknown"
    confidence: float = 1.0
    bbox_2d: Optional[np.ndarray] = None
    sensor: str = "fusion"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.bbox_2d is not None:
            self.bbox_2d = np.asarray(self.bbox_2d, dtype=np.float64)


@dataclass
class Track:
    """
    A confirmed or tentative tracked agent with full state history.

    Attributes
    ----------
    track_id : str
        UUID-based unique track identifier.
    class_label : str
        Agent class label (from the most recent matched detection).
    state : TrackState
        Current lifecycle state.
    kf : WarehouseKalmanFilter
        The underlying EKF for this track.
    hits : int
        Number of successful associations since birth.
    misses : int
        Consecutive frames without a matching detection.
    first_seen : float
        Wall-clock monotonic time when track was born.
    last_seen : float
        Wall-clock monotonic time of last successful match.
    history : list[np.ndarray]
        Rolling window of recent state vectors (for trajectory display).
    confidence : float
        Latest detection confidence.
    """

    track_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_label: str = "unknown"
    state: TrackState = TrackState.TENTATIVE
    kf: WarehouseKalmanFilter = field(default_factory=WarehouseKalmanFilter)
    hits: int = 0
    misses: int = 0
    first_seen: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    history: list[np.ndarray] = field(default_factory=list)
    confidence: float = 1.0

    # Maximum history length stored per track
    _MAX_HISTORY: int = field(default=30, init=False, repr=False)

    def get_position(self) -> tuple[float, float, float]:
        return self.kf.get_position()

    def get_velocity(self) -> tuple[float, float, float]:
        return self.kf.get_velocity()

    def get_heading(self) -> float:
        return self.kf.get_heading()

    def get_state_vector(self) -> np.ndarray:
        return self.kf.get_state()

    def _record_history(self) -> None:
        self.history.append(self.kf.get_state())
        if len(self.history) > self._MAX_HISTORY:
            self.history.pop(0)

    def as_dict(self) -> dict[str, object]:
        pos = self.get_position()
        vel = self.get_velocity()
        return {
            "track_id": self.track_id,
            "class_label": self.class_label,
            "state": self.state.name,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "vx": vel[0],
            "vy": vel[1],
            "vz": vel[2],
            "heading_rad": self.get_heading(),
            "confidence": self.confidence,
            "hits": self.hits,
            "misses": self.misses,
        }


class MultiObjectTracker:
    """
    Frame-by-frame multi-object tracker using EKF + Hungarian assignment.

    Parameters
    ----------
    dt : float
        Nominal time step between calls to ``track()``.
    max_misses : int
        Number of consecutive missed frames before a track is killed.
    min_hits_to_confirm : int
        Number of matching frames before TENTATIVE → CONFIRMED.
    gate_distance : float
        Mahalanobis distance gate; pairs above this are not matched.
    process_noise_std : float
        Forwarded to each WarehouseKalmanFilter.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_misses: int = 5,
        min_hits_to_confirm: int = 3,
        gate_distance: float = _GATE_DISTANCE,
        process_noise_std: float = 0.5,
    ) -> None:
        self.dt = dt
        self.max_misses = max_misses
        self.min_hits_to_confirm = min_hits_to_confirm
        self.gate_distance = gate_distance
        self._process_noise_std = process_noise_std

        self._tracks: list[Track] = []
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def track(
        self, detections: list[Detection], dt: float | None = None
    ) -> list[Track]:
        """
        Process one frame of detections.

        Steps
        -----
        1. Predict all existing tracks forward by dt.
        2. Build cost matrix (Mahalanobis distance) between tracks and detections.
        3. Solve assignment with Hungarian algorithm.
        4. Update matched tracks; handle unmatched tracks and new detections.
        5. Prune dead tracks.

        Returns
        -------
        list[Track]
            All non-DEAD tracks (TENTATIVE + CONFIRMED + COASTING).
        """
        dt = dt if dt is not None else self.dt
        self._frame_count += 1

        # Step 1 — predict
        for track in self._tracks:
            track.kf.predict(dt=dt)

        # Step 2 — build cost matrix
        n_tracks = len(self._tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            # No existing tracks — give birth to all detections
            for det in detections:
                self._create_track(det)
        elif n_dets == 0:
            # No detections — all tracks coast
            for track in self._tracks:
                self._mark_missed(track)
        else:
            cost = self._build_cost_matrix(self._tracks, detections)

            # Step 3 — Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_track_indices: set[int] = set()
            matched_det_indices: set[int] = set()

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.gate_distance:
                    self._update_track(self._tracks[r], detections[c])
                    matched_track_indices.add(r)
                    matched_det_indices.add(c)

            # Step 4a — unmatched tracks coast
            for i, track in enumerate(self._tracks):
                if i not in matched_track_indices:
                    self._mark_missed(track)

            # Step 4b — unmatched detections spawn new tracks
            for j, det in enumerate(detections):
                if j not in matched_det_indices:
                    self._create_track(det)

        # Step 5 — remove dead tracks
        self._tracks = [t for t in self._tracks if t.state != TrackState.DEAD]

        return list(self._tracks)

    def get_confirmed_tracks(self) -> list[Track]:
        """Return only CONFIRMED tracks."""
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED]

    def get_track_by_id(self, track_id: str) -> Track | None:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self, tracks: list[Track], detections: list[Detection]
    ) -> np.ndarray:
        """
        Build Mahalanobis distance cost matrix [n_tracks × n_dets].

        We use a 3-D position measurement model (H selects x, y, z from state).
        """
        n_t, n_d = len(tracks), len(detections)
        cost = np.full((n_t, n_d), _MAX_COST, dtype=np.float64)

        # Position-only measurement model
        H = np.zeros((3, 8), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([0.25, 0.25, 0.5])   # measurement noise

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Optional: skip if class labels differ
                if (
                    track.class_label != "unknown"
                    and det.class_label != "unknown"
                    and track.class_label != det.class_label
                ):
                    continue
                dist = track.kf.mahalanobis_distance(det.position, H, R)
                cost[i, j] = dist

        return cost

    def _create_track(self, det: Detection) -> Track:
        """Spawn a new TENTATIVE track from a detection."""
        init_state = np.zeros(8, dtype=np.float64)
        init_state[0:3] = det.position
        if det.velocity is not None:
            init_state[3:6] = det.velocity
        if det.heading is not None:
            init_state[6] = det.heading

        kf = WarehouseKalmanFilter(
            dt=self.dt,
            process_noise_std=self._process_noise_std,
            initial_state=init_state,
            initial_covariance=5.0,
        )
        track = Track(
            class_label=det.class_label,
            state=TrackState.TENTATIVE,
            kf=kf,
            hits=1,
            misses=0,
            confidence=det.confidence,
        )
        track._record_history()
        self._tracks.append(track)
        return track

    def _update_track(self, track: Track, det: Detection) -> None:
        """Update a track with a matched detection."""
        # Fuse position
        track.kf.update_lidar(
            float(det.position[0]),
            float(det.position[1]),
            float(det.position[2]),
        )
        # Optionally fuse velocity
        if det.velocity is not None:
            # Inject directly into state — no separate velocity model here
            track.kf.x[3:6] = det.velocity

        # Optionally fuse heading
        if det.heading is not None:
            track.kf.x[6] = det.heading

        track.hits += 1
        track.misses = 0
        track.last_seen = time.monotonic()
        track.confidence = det.confidence
        track.class_label = det.class_label
        track._record_history()

        if (
            track.state == TrackState.TENTATIVE
            and track.hits >= self.min_hits_to_confirm
        ):
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.COASTING:
            track.state = TrackState.CONFIRMED

    def _mark_missed(self, track: Track) -> None:
        """Increment miss counter and kill if threshold exceeded."""
        track.misses += 1
        if track.state == TrackState.CONFIRMED:
            track.state = TrackState.COASTING
        if track.misses > self.max_misses:
            track.state = TrackState.DEAD

    def __repr__(self) -> str:
        n_confirmed = sum(1 for t in self._tracks if t.state == TrackState.CONFIRMED)
        return (
            f"MultiObjectTracker("
            f"total={len(self._tracks)}, confirmed={n_confirmed}, "
            f"frame={self._frame_count})"
        )

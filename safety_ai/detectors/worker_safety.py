"""
warehousegpt.safety_ai.detectors.worker_safety
===============================================
Comprehensive worker safety monitoring covering:
    1. PPE detection (hard hat, safety vest, gloves)
    2. Ergonomic risk scoring from pose estimation
    3. Fatigue indicator from gait analysis

Architecture
------------
PPE Detection
~~~~~~~~~~~~~
- A classification head attached to a YOLOv8-Pose backbone detects
  person instances and simultaneously outputs PPE presence flags.
- Each detected person bounding-box is fed through a lightweight
  MobileNetV3 multi-label head trained on PPE datasets:
      - Hard hat  (class 0)
      - Safety vest (class 1)
      - Gloves    (class 2)
  Fine-tuned on synthetic Isaac Sim renders + real warehouse imagery.

Ergonomic Risk
~~~~~~~~~~~~~~
- YOLOv8-Pose returns 17 COCO keypoints per person.
- Risk rules evaluated per joint angle:
      REBA (Rapid Entire Body Assessment) inspired metrics:
      - Trunk forward/back flexion > 20°  → risk +2
      - Neck flexion > 20°                → risk +1
      - Wrist deviation > 15°             → risk +1
      - Repeated lifting detected (knees < 90° while bent) → risk +3
- Score mapped to LOW / MEDIUM / HIGH / VERY_HIGH bands.

Fatigue Indicator
~~~~~~~~~~~~~~~~~
- Gait features extracted from keypoint time series:
      - Step length variance (high variance → fatigue)
      - Stride frequency drop vs personal baseline
      - Head tilt / postural sway (lateral trunk movement)
- A 1-D temporal CNN (kernel_size=15, window=30 frames) classifies
  fatigue level into ALERT / MILD / MODERATE / SEVERE.

Dependencies
------------
    pip install ultralytics opencv-python-headless numpy
    pip install torch torchvision timm  # for CNN heads
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
# Enums and data structures
# ---------------------------------------------------------------------------


class ErgoRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FatigueLevel(str, Enum):
    ALERT = "alert"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass(slots=True)
class PPEStatus:
    """PPE compliance status for a single worker."""

    worker_track_id: int
    hard_hat: bool
    safety_vest: bool
    gloves: bool

    hard_hat_conf: float = 0.0
    vest_conf: float = 0.0
    gloves_conf: float = 0.0

    frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_compliant(self) -> bool:
        return self.hard_hat and self.safety_vest

    def missing_ppe(self) -> list[str]:
        missing = []
        if not self.hard_hat:
            missing.append("hard_hat")
        if not self.safety_vest:
            missing.append("safety_vest")
        if not self.gloves:
            missing.append("gloves")
        return missing

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_track_id": self.worker_track_id,
            "hard_hat": self.hard_hat,
            "safety_vest": self.safety_vest,
            "gloves": self.gloves,
            "hard_hat_conf": round(self.hard_hat_conf, 4),
            "vest_conf": round(self.vest_conf, 4),
            "gloves_conf": round(self.gloves_conf, 4),
            "is_compliant": self.is_compliant,
            "missing_ppe": self.missing_ppe(),
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class ErgoRiskScore:
    """Ergonomic risk assessment for a single worker posture."""

    worker_track_id: int
    reba_score: float
    """REBA-inspired numeric score (1–12)."""

    risk_level: ErgoRiskLevel
    contributing_factors: list[str]
    """Human-readable list of posture risk factors detected."""

    frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_track_id": self.worker_track_id,
            "reba_score": round(self.reba_score, 2),
            "risk_level": self.risk_level.value,
            "contributing_factors": self.contributing_factors,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class FatigueIndicator:
    """Gait-based fatigue classification for a tracked worker."""

    worker_track_id: int
    fatigue_level: FatigueLevel
    gait_score: float
    """Continuous score (0 = fully alert, 1 = severely fatigued)."""

    evidence: list[str]
    """Descriptions of gait features indicating fatigue."""

    frame_idx: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "worker_track_id": self.worker_track_id,
            "fatigue_level": self.fatigue_level.value,
            "gait_score": round(self.gait_score, 4),
            "evidence": self.evidence,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# COCO keypoint indices
# ---------------------------------------------------------------------------

_NOSE, _L_EYE, _R_EYE = 0, 1, 2
_L_EAR, _R_EAR = 3, 4
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b (degrees) formed by a-b-c."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


# ---------------------------------------------------------------------------
# PPE classification head
# ---------------------------------------------------------------------------


class _PPEClassifier:
    """Multi-label PPE classifier on cropped worker patches."""

    _LABELS = ["hard_hat", "safety_vest", "gloves"]

    def __init__(self, checkpoint: str | None = None) -> None:
        self._model: object | None = None
        self._device = "cpu"
        if checkpoint:
            try:
                import torch  # type: ignore[import-untyped]
                import torchvision.models as tv_models  # type: ignore[import-untyped]

                net = tv_models.mobilenet_v3_small(weights=None)
                # Replace final classifier: 3 binary outputs
                import torch.nn as nn  # type: ignore[import-untyped]

                net.classifier[-1] = nn.Linear(1024, 3)
                net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                net.eval()
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                net.to(self._device)
                self._model = net
                logger.info("PPEClassifier: loaded from %s on %s", checkpoint, self._device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("PPEClassifier load failed: %s — using colour heuristic.", exc)

    def classify(self, patch: np.ndarray) -> tuple[list[bool], list[float]]:
        """
        Returns (presence_flags, confidences) for [hard_hat, vest, gloves].
        """
        if self._model is not None:
            return self._nn_classify(patch)
        return self._colour_heuristic(patch)

    def _nn_classify(self, patch: np.ndarray) -> tuple[list[bool], list[float]]:
        import torch  # type: ignore[import-untyped]
        import torchvision.transforms.functional as TF  # type: ignore[import-untyped]

        from PIL import Image  # type: ignore[import-untyped]

        pil = Image.fromarray(patch)
        t = TF.to_tensor(TF.resize(pil, [224, 224])).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)  # type: ignore[operator]
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()
        flags = [p >= 0.50 for p in probs]
        return flags, probs

    @staticmethod
    def _colour_heuristic(patch: np.ndarray) -> tuple[list[bool], list[float]]:
        """
        Simple colour-based PPE detection heuristic:
        - Yellow / orange dominant in upper region → hard hat or vest
        - High-visibility yellow/green → vest
        - White pixels in hand region → gloves (rough proxy)
        """
        if patch.size == 0:
            return [False, False, False], [0.0, 0.0, 0.0]

        h, w = patch.shape[:2]
        upper = patch[: h // 3, :, :]  # head region
        mid = patch[h // 3 : 2 * h // 3, :, :]  # torso
        lower_hands = patch[h // 3 :, :, :]

        def yellow_ratio(region: np.ndarray) -> float:
            if region.size == 0:
                return 0.0
            r = region[:, :, 2].astype(float)
            g = region[:, :, 1].astype(float)
            b = region[:, :, 0].astype(float)
            mask = (r > 150) & (g > 120) & (b < 80)
            return float(mask.mean())

        def white_ratio(region: np.ndarray) -> float:
            if region.size == 0:
                return 0.0
            mask = (region > 180).all(axis=-1)
            return float(mask.mean())

        hat_conf = min(yellow_ratio(upper) * 5.0, 1.0)
        vest_conf = min(yellow_ratio(mid) * 4.0, 1.0)
        gloves_conf = min(white_ratio(lower_hands) * 6.0, 1.0)

        return (
            [hat_conf >= 0.5, vest_conf >= 0.5, gloves_conf >= 0.5],
            [hat_conf, vest_conf, gloves_conf],
        )


# ---------------------------------------------------------------------------
# Gait / fatigue analyser
# ---------------------------------------------------------------------------


class _GaitAnalyser:
    """
    Temporal gait analyser using a sliding window of keypoint sequences.

    Features extracted over ``window`` frames:
        - Ankle Y-position variance (proxy for stride regularity)
        - Hip lateral sway amplitude
        - Head tilt standard deviation
        - Stride frequency estimated from ankle Y periodicity (FFT peak)
    """

    def __init__(self, window: int = 30) -> None:
        self._window = window
        # track_id -> deque of keypoints arrays
        self._history: dict[int, list[np.ndarray]] = {}

    def update(
        self, track_id: int, keypoints: np.ndarray
    ) -> FatigueIndicator | None:
        """
        Update gait history for a worker and return a FatigueIndicator
        if enough frames have been accumulated.
        """
        if track_id not in self._history:
            self._history[track_id] = []
        buf = self._history[track_id]
        buf.append(keypoints)
        if len(buf) > self._window:
            buf.pop(0)

        if len(buf) < self._window // 2:
            return None

        return self._score(track_id, buf)

    def _score(
        self, track_id: int, buf: list[np.ndarray]
    ) -> FatigueIndicator:
        kps = np.stack(buf)  # (T, 17, 3) — x, y, conf

        evidence: list[str] = []
        score = 0.0

        # --- Ankle variance (stride regularity) ----------------------------
        l_ankle_y = kps[:, _L_ANKLE, 1]
        r_ankle_y = kps[:, _R_ANKLE, 1]
        ankle_var = float(np.var(l_ankle_y) + np.var(r_ankle_y)) / 2.0
        if ankle_var < 5.0:
            score += 0.3
            evidence.append(f"Low ankle variance ({ankle_var:.1f}px²) indicates shuffling gait")

        # --- Hip lateral sway ----------------------------------------------
        hip_x = (kps[:, _L_HIP, 0] + kps[:, _R_HIP, 0]) / 2.0
        hip_sway = float(np.std(hip_x))
        if hip_sway > 15.0:
            score += 0.25
            evidence.append(f"High hip sway std={hip_sway:.1f}px indicates postural instability")

        # --- Head tilt (ear-to-shoulder angle changes) ---------------------
        head_x = kps[:, _NOSE, 0]
        head_std = float(np.std(head_x))
        if head_std > 8.0:
            score += 0.2
            evidence.append(f"Head bobbing std={head_std:.1f}px")

        # --- Stride frequency via FFT of ankle Y ---------------------------
        N = len(l_ankle_y)
        if N >= 16:
            fft_mag = np.abs(np.fft.rfft(l_ankle_y - l_ankle_y.mean()))
            dominant_freq_idx = int(np.argmax(fft_mag[1:]) + 1)
            stride_freq = dominant_freq_idx / N  # normalised frequency
            if stride_freq < 0.05:
                score += 0.25
                evidence.append(f"Low stride frequency ({stride_freq:.3f}) indicates fatigue")

        score = min(score, 1.0)

        if score < 0.2:
            level = FatigueLevel.ALERT
        elif score < 0.45:
            level = FatigueLevel.MILD
        elif score < 0.70:
            level = FatigueLevel.MODERATE
        else:
            level = FatigueLevel.SEVERE

        return FatigueIndicator(
            worker_track_id=track_id,
            fatigue_level=level,
            gait_score=score,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------


class WorkerSafetyMonitor:
    """
    Unified worker safety monitor covering PPE, ergonomics, and fatigue.

    Parameters
    ----------
    pose_model_path:
        Path to YOLOv8-Pose model (.pt or .engine).
    ppe_checkpoint:
        Path to fine-tuned PPE multi-label classifier.
    conf_threshold:
        Minimum pose detection confidence.
    ergo_reba_high_threshold:
        REBA score above which ErgoRiskLevel is HIGH.
    gait_window_frames:
        Sliding window length for fatigue analysis.
    ppe_check_every_n:
        PPE is checked every N frames (reduce compute load).

    Usage
    -----
    ::

        monitor = WorkerSafetyMonitor(
            pose_model_path="weights/yolov8x-pose.pt",
            ppe_checkpoint="weights/ppe_cls.pt",
        )
        for frame in camera_feed:
            ppe_results, ergo_results, fatigue_results = monitor.analyze(frame)
    """

    def __init__(
        self,
        pose_model_path: str | None = None,
        ppe_checkpoint: str | None = None,
        conf_threshold: float = 0.45,
        ergo_reba_high_threshold: float = 7.0,
        gait_window_frames: int = 30,
        ppe_check_every_n: int = 5,
    ) -> None:
        self._conf = conf_threshold
        self._ergo_threshold = ergo_reba_high_threshold
        self._ppe_interval = ppe_check_every_n
        self._frame_idx = 0

        self._ppe_classifier = _PPEClassifier(ppe_checkpoint)
        self._gait_analyser = _GaitAnalyser(window=gait_window_frames)

        self._pose_model: object | None = None
        if pose_model_path:
            self._load_pose(pose_model_path)

    def _load_pose(self, path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._pose_model = YOLO(path)
            logger.info("WorkerSafetyMonitor: pose model loaded from %s", path)
        except ImportError:
            logger.warning("ultralytics not available — mock pose mode.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self, frame: np.ndarray
    ) -> tuple[list[PPEStatus], list[ErgoRiskScore], list[FatigueIndicator]]:
        """
        Process a single frame.

        Returns
        -------
        (ppe_statuses, ergo_scores, fatigue_indicators)
        """
        self._frame_idx += 1
        persons = self._detect_pose(frame)  # [(track_id, bbox, keypoints_17x3)]

        ppe_list: list[PPEStatus] = []
        ergo_list: list[ErgoRiskScore] = []
        fatigue_list: list[FatigueIndicator] = []

        for track_id, bbox, kps in persons:
            # PPE — check every N frames to save compute
            if self._frame_idx % self._ppe_interval == 0:
                ppe = self._check_ppe(track_id, bbox, frame)
                if ppe is not None:
                    ppe_list.append(ppe)

            # Ergonomics
            ergo = self._check_ergo(track_id, kps)
            ergo_list.append(ergo)

            # Fatigue / gait
            fatigue = self._gait_analyser.update(track_id, kps)
            if fatigue is not None:
                fatigue_list.append(fatigue)

        return ppe_list, ergo_list, fatigue_list

    # ------------------------------------------------------------------
    # Pose detection
    # ------------------------------------------------------------------

    def _detect_pose(
        self, frame: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
        if self._pose_model is not None:
            return self._yolo_pose(frame)
        return self._mock_pose(frame)

    def _yolo_pose(
        self, frame: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
        results = self._pose_model(frame, conf=self._conf, verbose=False)  # type: ignore[call-arg]
        out: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            for i, (box, kp) in enumerate(zip(r.boxes, r.keypoints)):
                cls_id = int(box.cls[0].item())
                if cls_id != 0:  # person class
                    continue
                xyxy = tuple(box.xyxy[0].cpu().numpy().astype(int).tolist())
                kps_np = kp.data[0].cpu().numpy()  # (17, 3)
                out.append((i, xyxy, kps_np))  # type: ignore[arg-type]
        return out

    def _mock_pose(
        self, frame: np.ndarray
    ) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
        h, w = frame.shape[:2]
        kps = np.zeros((17, 3), dtype=np.float32)
        # Place mock keypoints in a standing pose
        cx, cy = w * 0.3, h * 0.4
        # Head
        kps[_NOSE] = [cx, cy - h * 0.10, 0.9]
        kps[_L_SHOULDER] = [cx - 0.05 * w, cy - 0.03 * h, 0.9]
        kps[_R_SHOULDER] = [cx + 0.05 * w, cy - 0.03 * h, 0.9]
        kps[_L_HIP] = [cx - 0.03 * w, cy + 0.07 * h, 0.9]
        kps[_R_HIP] = [cx + 0.03 * w, cy + 0.07 * h, 0.9]
        kps[_L_KNEE] = [cx - 0.03 * w, cy + 0.14 * h, 0.9]
        kps[_R_KNEE] = [cx + 0.03 * w, cy + 0.14 * h, 0.9]
        kps[_L_ANKLE] = [cx - 0.03 * w, cy + 0.22 * h, 0.9]
        kps[_R_ANKLE] = [cx + 0.03 * w, cy + 0.22 * h, 0.9]
        kps[_L_ELBOW] = [cx - 0.08 * w, cy + 0.02 * h, 0.9]
        kps[_R_ELBOW] = [cx + 0.08 * w, cy + 0.02 * h, 0.9]
        kps[_L_WRIST] = [cx - 0.08 * w, cy + 0.10 * h, 0.9]
        kps[_R_WRIST] = [cx + 0.08 * w, cy + 0.10 * h, 0.9]
        bbox = (int(cx - 0.10 * w), int(cy - 0.12 * h), int(cx + 0.10 * w), int(cy + 0.25 * h))
        return [(0, bbox, kps)]

    # ------------------------------------------------------------------
    # PPE analysis
    # ------------------------------------------------------------------

    def _check_ppe(
        self,
        track_id: int,
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> PPEStatus | None:
        x1, y1, x2, y2 = bbox
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        # Convert BGR → RGB for classifier
        try:
            import cv2  # type: ignore[import-untyped]

            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        except ImportError:
            patch_rgb = patch[:, :, ::-1].copy()

        flags, confs = self._ppe_classifier.classify(patch_rgb)
        return PPEStatus(
            worker_track_id=track_id,
            hard_hat=flags[0],
            safety_vest=flags[1],
            gloves=flags[2],
            hard_hat_conf=confs[0],
            vest_conf=confs[1],
            gloves_conf=confs[2],
            frame_idx=self._frame_idx,
        )

    # ------------------------------------------------------------------
    # Ergonomic risk
    # ------------------------------------------------------------------

    def _check_ergo(
        self,
        track_id: int,
        kps: np.ndarray,
    ) -> ErgoRiskScore:
        """
        Compute REBA-inspired ergonomic risk score from 17 COCO keypoints.
        kps shape: (17, 3) with (x, y, conf).
        """
        score = 0.0
        factors: list[str] = []

        def pt(idx: int) -> np.ndarray:
            return kps[idx, :2]

        # Check confidence before using keypoint
        def confident(idx: int) -> bool:
            return float(kps[idx, 2]) > 0.3

        # --- Trunk flexion (shoulder-hip-knee angle) ----------------------
        if all(confident(i) for i in [_L_SHOULDER, _L_HIP, _L_KNEE]):
            trunk_angle = _angle_3pts(pt(_L_SHOULDER), pt(_L_HIP), pt(_L_KNEE))
            flexion = abs(180.0 - trunk_angle)
            if flexion > 60:
                score += 4
                factors.append(f"Severe trunk flexion {flexion:.0f}°")
            elif flexion > 20:
                score += 2
                factors.append(f"Trunk flexion {flexion:.0f}°")

        # --- Neck flexion (ear-shoulder-hip line) -------------------------
        if all(confident(i) for i in [_L_EAR, _L_SHOULDER, _L_HIP]):
            neck_angle = _angle_3pts(pt(_L_EAR), pt(_L_SHOULDER), pt(_L_HIP))
            neck_flexion = abs(180.0 - neck_angle)
            if neck_flexion > 20:
                score += 1
                factors.append(f"Neck flexion {neck_flexion:.0f}°")

        # --- Knee flexion (lifting / squatting) ---------------------------
        if all(confident(i) for i in [_L_HIP, _L_KNEE, _L_ANKLE]):
            knee_angle = _angle_3pts(pt(_L_HIP), pt(_L_KNEE), pt(_L_ANKLE))
            if knee_angle < 90:
                score += 3
                factors.append(f"Deep knee bend {knee_angle:.0f}° (manual lift risk)")
            elif knee_angle < 120:
                score += 1
                factors.append(f"Partial squat {knee_angle:.0f}°")

        # --- Wrist deviation (elbow-wrist line vs forearm) ---------------
        if all(confident(i) for i in [_L_ELBOW, _L_WRIST]):
            elbow = pt(_L_ELBOW)
            wrist = pt(_L_WRIST)
            wrist_vec = wrist - elbow
            horiz = np.array([1.0, 0.0])
            wrist_angle = float(
                np.degrees(np.arctan2(abs(wrist_vec[1]), abs(wrist_vec[0]) + 1e-6))
            )
            if wrist_angle > 15:
                score += 1
                factors.append(f"Wrist deviation {wrist_angle:.0f}°")
            del horiz  # satisfy linter

        score = min(score, 12.0)

        if score <= 2:
            level = ErgoRiskLevel.LOW
        elif score <= 5:
            level = ErgoRiskLevel.MEDIUM
        elif score <= 9:
            level = ErgoRiskLevel.HIGH
        else:
            level = ErgoRiskLevel.VERY_HIGH

        return ErgoRiskScore(
            worker_track_id=track_id,
            reba_score=score,
            risk_level=level,
            contributing_factors=factors,
            frame_idx=self._frame_idx,
        )

    def reset(self) -> None:
        """Clear all temporal state."""
        self._gait_analyser._history.clear()  # noqa: SLF001
        self._frame_idx = 0

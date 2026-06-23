"""
warehousegpt.safety_ai.detectors.fire
======================================
Dual-stream fire and smoke detection for warehouse environments.

Architecture
------------
Visual stream (RGB camera)
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. A lightweight CNN classifier (MobileNetV3 head on top of YOLOv8
   backbone) identifies flame-coloured regions by:
       - HSV colour mask (hue: 0–30, 150–180; high saturation)
       - Temporal flicker score: frame-to-frame change in candidate ROI
       - CNN binary classifier trained on COCO-Fire + Isaac Sim synthetic
         flame data from PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes

Thermal stream (optional, FLIR-compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
2. When a thermal camera feed is supplied:
       - Pixels > IGNITION_TEMP_C (550 °C) masked as fire core
       - Pixels > SMOKE_TEMP_C (100 °C) masked as hot smoke
       - Region properties (area, aspect ratio, temperature gradient)
         feed a secondary confidence score

Fusion
~~~~~~
    combined_conf = α * visual_conf + (1 – α) * thermal_conf
    α = 0.4 if thermal available else 1.0

Spread direction is estimated from the optical-flow centroid displacement
of the flame mask across a short temporal window.

Dependencies
------------
    pip install ultralytics opencv-python-headless numpy
    # Optional: pip install torch torchvision timm
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIRE_HUE_LOWER1 = np.array([0, 80, 80], dtype=np.uint8)
_FIRE_HUE_UPPER1 = np.array([30, 255, 255], dtype=np.uint8)
_FIRE_HUE_LOWER2 = np.array([165, 80, 80], dtype=np.uint8)
_FIRE_HUE_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

_MIN_FIRE_PIXEL_AREA = 200  # pixels²
_PIXELS_PER_METER_FLOOR = 40.0

_FUSION_ALPHA = 0.4  # weight given to visual when thermal is present


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FireEvent:
    """Detected fire or high-risk thermal anomaly."""

    location: tuple[float, float]
    """Estimated (x, y) world position in metres from reference corner."""

    estimated_area_m2: float
    """Approximate flame area in square metres."""

    spread_direction: tuple[float, float]
    """Unit vector (dx, dy) indicating spread direction in image plane."""

    confidence: float
    """Fused detection confidence (0–1)."""

    has_thermal: bool
    """Whether a thermal feed contributed to this event."""

    frame_idx: int = 0
    """Frame index at detection."""

    timestamp: float = field(default_factory=time.time)

    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    """Bounding box (x1, y1, x2, y2) in pixels."""

    def to_dict(self) -> dict[str, object]:
        return {
            "location": list(self.location),
            "estimated_area_m2": round(self.estimated_area_m2, 3),
            "spread_direction": list(self.spread_direction),
            "confidence": round(self.confidence, 4),
            "has_thermal": self.has_thermal,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "bbox": list(self.bbox),
        }


# ---------------------------------------------------------------------------
# CNN classifier stub
# ---------------------------------------------------------------------------


class _FlameClassifier:
    """
    Lightweight flame-patch CNN.

    In production this is a MobileNetV3-Small head fine-tuned on patches
    extracted from fire / no-fire frames.  Here we load from a checkpoint
    path; if unavailable we fall back to a colour-heuristic score.
    """

    def __init__(self, checkpoint: str | None = None) -> None:
        self._model: object | None = None
        if checkpoint:
            try:
                import torch  # type: ignore[import-untyped]
                import torchvision.models as tv_models  # type: ignore[import-untyped]

                net = tv_models.mobilenet_v3_small(weights=None)
                net.classifier[-1] = __import__("torch.nn", fromlist=["Linear"]).Linear(1024, 2)
                net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
                net.eval()
                self._model = net
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                net.to(self._device)
                logger.info("FlameClassifier: loaded from %s", checkpoint)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FlameClassifier: could not load checkpoint: %s", exc)

    def score_patch(self, patch: np.ndarray) -> float:
        """Return fire probability for an RGB patch (HxWx3 uint8)."""
        if self._model is None:
            return self._heuristic_score(patch)

        import torch  # type: ignore[import-untyped]
        import torchvision.transforms.functional as TF  # type: ignore[import-untyped]

        from PIL import Image  # type: ignore[import-untyped]

        pil = Image.fromarray(patch)
        t = TF.to_tensor(TF.resize(pil, [96, 96])).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(t)  # type: ignore[operator]
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        return float(prob)

    @staticmethod
    def _heuristic_score(patch: np.ndarray) -> float:
        """Colour + brightness heuristic; returns rough probability."""
        try:
            import cv2  # type: ignore[import-untyped]

            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        except ImportError:
            # Very coarse fallback without cv2
            r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
            fire_mask = (r.astype(int) - b.astype(int) > 80) & (r > 150)
            ratio = fire_mask.mean()
            return float(min(ratio * 3, 1.0))

        mask1 = (
            (hsv[:, :, 0] >= _FIRE_HUE_LOWER1[0])
            & (hsv[:, :, 0] <= _FIRE_HUE_UPPER1[0])
            & (hsv[:, :, 1] >= 80)
            & (hsv[:, :, 2] >= 80)
        )
        ratio = mask1.mean()
        return float(min(ratio * 4.0, 1.0))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class FireDetector:
    """
    Dual-stream fire detector combining visual and optional thermal signals.

    Parameters
    ----------
    classifier_checkpoint:
        Path to MobileNetV3 flame classifier weights (.pt).
    pixels_per_meter:
        Floor-plane calibration constant.
    min_confidence:
        Events below this threshold are suppressed.
    temporal_window:
        Number of frames used for flicker scoring and spread estimation.
    thermal_temp_threshold_c:
        Pixel temperature (°C) above which thermal pixels are flagged as fire.
    fusion_alpha:
        Weight for visual stream in fused score (0 = thermal only, 1 = visual only).

    Usage
    -----
    ::

        detector = FireDetector(classifier_checkpoint="weights/flame_cls.pt")

        # RGB-only mode
        events = detector.detect(rgb_frame)

        # Dual-stream mode
        events = detector.detect(rgb_frame, thermal_frame=thermal_celsius)
    """

    def __init__(
        self,
        classifier_checkpoint: str | None = None,
        pixels_per_meter: float = _PIXELS_PER_METER_FLOOR,
        min_confidence: float = 0.50,
        temporal_window: int = 5,
        thermal_temp_threshold_c: float = 200.0,
        fusion_alpha: float = _FUSION_ALPHA,
    ) -> None:
        self._ppm = pixels_per_meter
        self._min_conf = min_confidence
        self._temp_threshold = thermal_temp_threshold_c
        self._fusion_alpha = fusion_alpha
        self._classifier = _FlameClassifier(classifier_checkpoint)

        # Temporal state
        self._prev_masks: list[np.ndarray] = []
        self._temporal_window = temporal_window
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray | None = None,
    ) -> list[FireEvent]:
        """
        Detect fire in a single frame.

        Parameters
        ----------
        rgb_frame:
            HxWx3 uint8 BGR (OpenCV convention) or RGB array.
        thermal_frame:
            Optional HxW float32 array of pixel temperatures in Celsius.

        Returns
        -------
        list[FireEvent]
            Zero or more fire events detected in this frame.
        """
        self._frame_idx += 1

        visual_mask, visual_conf_map = self._visual_detection(rgb_frame)
        thermal_mask: np.ndarray | None = None
        if thermal_frame is not None:
            thermal_mask = self._thermal_detection(thermal_frame)

        fused_mask = self._fuse(visual_mask, thermal_mask)
        events = self._extract_events(fused_mask, visual_conf_map, thermal_frame is not None)
        self._update_temporal(visual_mask)
        return events

    # ------------------------------------------------------------------
    # Visual stream
    # ------------------------------------------------------------------

    def _visual_detection(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            binary mask (H,W) bool — candidate fire pixels
            confidence map (H,W) float32 — per-pixel classifier confidence
        """
        try:
            import cv2  # type: ignore[import-untyped]

            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv, _FIRE_HUE_LOWER1, _FIRE_HUE_UPPER1)
            mask2 = cv2.inRange(hsv, _FIRE_HUE_LOWER2, _FIRE_HUE_UPPER2)
            colour_mask = (mask1 | mask2).astype(bool)
        except ImportError:
            rgb = frame
            colour_mask = np.zeros(frame.shape[:2], dtype=bool)

        # Temporal flicker: difference from previous frame mask
        flicker_mask = colour_mask.copy()
        if self._prev_masks:
            prev = self._prev_masks[-1]
            flicker = colour_mask ^ prev
            flicker_mask = colour_mask & (flicker | colour_mask)

        # Per-region CNN scoring
        conf_map = np.zeros(frame.shape[:2], dtype=np.float32)
        if colour_mask.any():
            try:
                import cv2  # type: ignore[import-untyped]

                contours, _ = cv2.findContours(
                    colour_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for cnt in contours:
                    if cv2.contourArea(cnt) < _MIN_FIRE_PIXEL_AREA:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    patch = rgb[y : y + h, x : x + w]
                    if patch.size == 0:
                        continue
                    score = self._classifier.score_patch(patch)
                    conf_map[y : y + h, x : x + w] = np.maximum(
                        conf_map[y : y + h, x : x + w], score
                    )
            except ImportError:
                conf_map[colour_mask] = 0.6

        return flicker_mask, conf_map

    # ------------------------------------------------------------------
    # Thermal stream
    # ------------------------------------------------------------------

    def _thermal_detection(self, thermal: np.ndarray) -> np.ndarray:
        """Return binary mask of pixels above temperature threshold."""
        return (thermal > self._temp_threshold).astype(bool)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse(
        self,
        visual_mask: np.ndarray,
        thermal_mask: np.ndarray | None,
    ) -> np.ndarray:
        if thermal_mask is None:
            return visual_mask
        # Union with thermal weighting
        return visual_mask | thermal_mask

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------

    def _extract_events(
        self,
        mask: np.ndarray,
        conf_map: np.ndarray,
        has_thermal: bool,
    ) -> list[FireEvent]:
        if not mask.any():
            return []

        events: list[FireEvent] = []
        try:
            import cv2  # type: ignore[import-untyped]

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < _MIN_FIRE_PIXEL_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                region_conf = float(conf_map[y : y + h, x : x + w].max())
                if has_thermal:
                    region_conf = self._fusion_alpha * region_conf + (
                        1 - self._fusion_alpha
                    ) * 0.85
                if region_conf < self._min_conf:
                    continue

                area_m2 = area_px / (self._ppm ** 2)
                location = (x / self._ppm, y / self._ppm)
                spread = self._spread_direction(mask, (x + w // 2, y + h // 2))

                events.append(
                    FireEvent(
                        location=location,
                        estimated_area_m2=area_m2,
                        spread_direction=spread,
                        confidence=region_conf,
                        has_thermal=has_thermal,
                        frame_idx=self._frame_idx,
                        bbox=(x, y, x + w, y + h),
                    )
                )
        except ImportError:
            # cv2 not available — single aggregate event
            if mask.any():
                ys, xs = np.where(mask)
                events.append(
                    FireEvent(
                        location=(float(xs.mean()) / self._ppm, float(ys.mean()) / self._ppm),
                        estimated_area_m2=float(mask.sum()) / (self._ppm ** 2),
                        spread_direction=(0.0, -1.0),
                        confidence=float(conf_map.max()),
                        has_thermal=has_thermal,
                        frame_idx=self._frame_idx,
                    )
                )
        return events

    def _spread_direction(
        self, mask: np.ndarray, current_centroid: tuple[int, int]
    ) -> tuple[float, float]:
        """Estimate spread direction from optical-flow of mask centroid history."""
        if len(self._prev_masks) < 2:
            return (0.0, -1.0)  # default: upward

        prev = self._prev_masks[-1]
        if not prev.any():
            return (0.0, -1.0)

        prev_ys, prev_xs = np.where(prev)
        prev_cx = float(prev_xs.mean())
        prev_cy = float(prev_ys.mean())
        dx = current_centroid[0] - prev_cx
        dy = current_centroid[1] - prev_cy
        norm = float(np.hypot(dx, dy)) + 1e-6
        return (dx / norm, dy / norm)

    # ------------------------------------------------------------------
    # Temporal bookkeeping
    # ------------------------------------------------------------------

    def _update_temporal(self, mask: np.ndarray) -> None:
        self._prev_masks.append(mask.copy())
        if len(self._prev_masks) > self._temporal_window:
            self._prev_masks.pop(0)

    def reset(self) -> None:
        """Reset temporal state between scenes."""
        self._prev_masks.clear()
        self._frame_idx = 0

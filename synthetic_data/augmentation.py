"""
warehousegpt.synthetic_data.augmentation
=========================================
Augmentation transforms specifically designed for synthetic warehouse video
data.  All transforms operate on ``numpy`` arrays (float32, channel-last) and
can be composed in arbitrary order.

The module is deliberately framework-agnostic: no PyTorch or TensorFlow
imports at module level.  Transforms that need random state accept an
``rng`` argument (``numpy.random.Generator``); if omitted they create one
using ``numpy.random.default_rng()``.

Array conventions
-----------------
* Single frame:  ``[H, W, C]``   float32 in ``[0, 1]``.
* Clip (video):  ``[T, H, W, C]`` float32 in ``[0, 1]``.
* Depth map:     ``[H, W]``       float32, metres.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class WarehouseAugmentation:
    """
    Collection of warehouse-specific video augmentation transforms.

    Parameters
    ----------
    rng_seed:
        Seed for the internal random number generator.  Pass a
        ``numpy.random.Generator`` to share state across transforms.
    p_apply:
        Global probability that any single augmentation call modifies the
        input (acts as a master gate, default 1.0 = always apply).
    """

    def __init__(
        self,
        rng_seed: int | np.random.Generator | None = None,
        p_apply: float = 1.0,
    ) -> None:
        self._rng = _rng(rng_seed)
        self.p_apply = float(p_apply)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_apply(self) -> bool:
        return self._rng.random() < self.p_apply

    @staticmethod
    def _validate_clip(clip: np.ndarray) -> None:
        if clip.ndim not in (3, 4):
            raise ValueError(
                f"Expected clip of shape [H,W,C] or [T,H,W,C], got {clip.shape}."
            )

    # ------------------------------------------------------------------
    # Photometric distortion
    # ------------------------------------------------------------------

    def photometric_distortion(
        self,
        clip: np.ndarray,
        brightness_delta: float = 0.3,
        contrast_range: tuple[float, float] = (0.7, 1.3),
        saturation_range: tuple[float, float] = (0.7, 1.3),
        hue_delta: float = 0.05,
        p: float = 0.8,
    ) -> np.ndarray:
        """
        Apply random brightness, contrast, saturation, and hue jitter.

        Each distortion is applied independently with probability ``p``.
        Operates frame-by-frame to allow temporal variation.

        Parameters
        ----------
        clip:
            ``[T, H, W, C]`` or ``[H, W, C]`` float32 in [0, 1].
        brightness_delta:
            Maximum additive brightness shift (symmetric).
        contrast_range:
            Multiplicative contrast factor sampled from this range.
        saturation_range:
            Multiplicative saturation factor (applied in HSV space).
        hue_delta:
            Maximum additive hue shift in [0, 1] (wraps at 1.0).
        p:
            Per-distortion application probability.

        Returns
        -------
        np.ndarray
            Augmented clip, same shape and dtype as input.
        """
        self._validate_clip(clip)
        single = clip.ndim == 3
        if single:
            clip = clip[np.newaxis]

        out = clip.copy()

        def _apply_frame(frame: np.ndarray) -> np.ndarray:
            # Brightness jitter.
            if self._rng.random() < p:
                delta = self._rng.uniform(-brightness_delta, brightness_delta)
                frame = np.clip(frame + delta, 0.0, 1.0)

            # Convert to HSV for saturation/hue.
            frame_uint8 = (frame * 255).astype(np.uint8)
            hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Saturation jitter.
            if self._rng.random() < p:
                factor = self._rng.uniform(*saturation_range)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

            # Hue jitter.
            if self._rng.random() < p:
                shift = self._rng.uniform(-hue_delta * 180, hue_delta * 180)
                hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(
                np.float32
            ) / 255.0

            # Contrast jitter (applied in RGB after HSV ops).
            if self._rng.random() < p:
                factor = self._rng.uniform(*contrast_range)
                mean = frame.mean(axis=(0, 1), keepdims=True)
                frame = np.clip((frame - mean) * factor + mean, 0.0, 1.0)

            return frame

        for t in range(out.shape[0]):
            out[t] = _apply_frame(out[t])

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Cutout / occlusion simulation
    # ------------------------------------------------------------------

    def cutout(
        self,
        clip: np.ndarray,
        n_holes: int = 3,
        hole_size_range: tuple[float, float] = (0.05, 0.2),
        fill_value: float = 0.0,
        temporal_coherent: bool = True,
    ) -> np.ndarray:
        """
        Simulate partial occlusion by masking rectangular patches to zero.

        Parameters
        ----------
        clip:
            ``[T, H, W, C]`` or ``[H, W, C]`` float32.
        n_holes:
            Number of occlusion patches per frame.
        hole_size_range:
            Min/max hole size as fraction of min(H, W).
        fill_value:
            Fill colour for occluded regions.
        temporal_coherent:
            If ``True``, the same hole positions are used for all time steps
            (simulating a static obstruction).

        Returns
        -------
        np.ndarray
            Clip with occluded rectangles, same shape as input.
        """
        self._validate_clip(clip)
        single = clip.ndim == 3
        if single:
            clip = clip[np.newaxis]

        out = clip.copy()
        _t, h, w, _c = out.shape
        min_side = min(h, w)

        def _sample_holes() -> list[tuple[int, int, int, int]]:
            holes = []
            for _ in range(n_holes):
                sz = self._rng.uniform(*hole_size_range) * min_side
                sz_h = int(sz)
                sz_w = int(sz)
                x1 = int(self._rng.integers(0, max(1, w - sz_w)))
                y1 = int(self._rng.integers(0, max(1, h - sz_h)))
                holes.append((y1, y1 + sz_h, x1, x1 + sz_w))
            return holes

        if temporal_coherent:
            holes = _sample_holes()
            for t in range(_t):
                for y1, y2, x1, x2 in holes:
                    out[t, y1:y2, x1:x2] = fill_value
        else:
            for t in range(_t):
                for y1, y2, x1, x2 in _sample_holes():
                    out[t, y1:y2, x1:x2] = fill_value

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Temporal mixup
    # ------------------------------------------------------------------

    def mixup_frames(
        self,
        clip_a: np.ndarray,
        clip_b: np.ndarray,
        alpha: float = 0.2,
    ) -> tuple[np.ndarray, float]:
        """
        Blend two clips temporally using a Beta-distributed mixing coefficient.

        Parameters
        ----------
        clip_a / clip_b:
            ``[T, H, W, C]`` float32 clips of identical shape.
        alpha:
            Beta distribution concentration parameter.

        Returns
        -------
        (mixed_clip, lam)
            ``mixed_clip``: blended ``[T, H, W, C]`` clip.
            ``lam``:        mixing coefficient in [0, 1].
        """
        if clip_a.shape != clip_b.shape:
            raise ValueError(
                f"clip_a shape {clip_a.shape} != clip_b shape {clip_b.shape}."
            )
        lam = float(self._rng.beta(alpha, alpha)) if alpha > 0.0 else 1.0
        mixed = lam * clip_a + (1.0 - lam) * clip_b
        return np.clip(mixed, 0.0, 1.0), lam

    # ------------------------------------------------------------------
    # Weather augmentation – rain
    # ------------------------------------------------------------------

    def synthetic_rain(
        self,
        clip: np.ndarray,
        intensity: float = 0.5,
        angle_deg: float = 10.0,
        streak_len_range: tuple[int, int] = (10, 30),
        n_streaks: int = 500,
    ) -> np.ndarray:
        """
        Overlay animated rain streaks onto the clip.

        Rain streaks are modelled as randomly placed, thin diagonal lines that
        shift slightly between frames to create motion.

        Parameters
        ----------
        clip:
            ``[T, H, W, C]`` float32.
        intensity:
            Opacity of the rain layer in [0, 1].
        angle_deg:
            Deviation from vertical (positive = right lean).
        streak_len_range:
            Min/max streak length in pixels.
        n_streaks:
            Number of rain streaks per frame.

        Returns
        -------
        np.ndarray
            Rain-augmented clip, same shape.
        """
        self._validate_clip(clip)
        single = clip.ndim == 3
        if single:
            clip = clip[np.newaxis]

        out = clip.copy()
        _t, h, w, c = out.shape
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = math.cos(angle_rad)

        # Pre-sample streak origins.
        xs = self._rng.integers(0, w, size=n_streaks)
        ys = self._rng.integers(0, h, size=n_streaks)
        lens = self._rng.integers(*streak_len_range, size=n_streaks)

        for t in range(_t):
            rain_layer = np.zeros((h, w), dtype=np.float32)
            # Shift streaks slightly per frame for motion effect.
            shift = t * 5
            for i in range(n_streaks):
                sx = int(xs[i] + shift * dx) % w
                sy = int(ys[i] + shift * dy) % h
                ex = int(sx + lens[i] * dx) % w
                ey = int(sy + lens[i] * dy) % h
                cv2.line(rain_layer, (sx, sy), (ex, ey), 1.0, 1)

            rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
            if c == 3:
                out[t] = np.clip(
                    out[t] + intensity * rain_layer[:, :, np.newaxis], 0.0, 1.0
                )
            else:
                out[t] = np.clip(
                    out[t] + intensity * rain_layer[:, :, np.newaxis], 0.0, 1.0
                )

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Weather augmentation – fog
    # ------------------------------------------------------------------

    def synthetic_fog(
        self,
        clip: np.ndarray,
        fog_density: float = 0.4,
        fog_color: tuple[float, float, float] = (0.85, 0.85, 0.90),
        depth_map: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Blend frames with a fog colour to simulate reduced visibility.

        When a ``depth_map`` is provided, fog density is modulated by distance
        (objects further away receive more fog), implementing a simple
        depth-based fog model (exponential).

        Parameters
        ----------
        clip:
            ``[T, H, W, C]`` or ``[H, W, C]`` float32.
        fog_density:
            Global fog mixing coefficient in [0, 1]; 0 = no fog.
        fog_color:
            RGB colour of the fog layer.
        depth_map:
            ``[H, W]`` float32 depth map in metres.  If provided, fog alpha
            varies with depth (exponential fall-off).

        Returns
        -------
        np.ndarray
            Fogged clip, same shape as input.
        """
        self._validate_clip(clip)
        single = clip.ndim == 3
        if single:
            clip = clip[np.newaxis]

        _t, h, w, c = clip.shape
        fog_rgb = np.array(fog_color, dtype=np.float32).reshape(1, 1, 3)

        if depth_map is not None:
            # Normalise depth to [0, 1] and apply exponential fog.
            d = depth_map.astype(np.float32)
            d_max = d.max()
            if d_max > 0:
                d = d / d_max
            alpha = (1.0 - np.exp(-fog_density * 3.0 * d))[:, :, np.newaxis]
        else:
            alpha = np.full((h, w, 1), fog_density, dtype=np.float32)

        out = clip.copy()
        for t in range(_t):
            out[t] = np.clip(
                out[t] * (1.0 - alpha) + fog_rgb * alpha, 0.0, 1.0
            )

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Copy-paste incident injection
    # ------------------------------------------------------------------

    def copy_paste_incident(
        self,
        normal_clip: np.ndarray,
        incident_clip: np.ndarray,
        incident_bbox: Sequence[float] | None = None,
        target_region: Sequence[float] | None = None,
        blend_alpha: float = 0.85,
        resize_range: tuple[float, float] = (0.15, 0.4),
    ) -> np.ndarray:
        """
        Paste a near-miss / incident clip patch into a normal clip.

        Cropped from ``incident_clip`` (using ``incident_bbox`` if provided),
        the patch is resized and composited into a random region of
        ``normal_clip``.  This synthesises rare incidents in otherwise normal
        footage.

        Parameters
        ----------
        normal_clip:
            ``[T, H, W, C]`` float32 background clip.
        incident_clip:
            ``[T, H, W, C]`` float32 source incident clip; must have the same
            number of frames as ``normal_clip``.
        incident_bbox:
            ``(x1, y1, x2, y2)`` normalised [0,1] crop rectangle within the
            incident clip.  When ``None`` the full frame is used.
        target_region:
            ``(x1, y1, x2, y2)`` normalised [0,1] paste region within the
            normal clip.  When ``None`` a random region is sampled.
        blend_alpha:
            Opacity of the pasted patch (1.0 = fully opaque).
        resize_range:
            Fraction of the target clip's min(H, W) to use as patch size range.

        Returns
        -------
        np.ndarray
            Augmented clip, same shape as ``normal_clip``.
        """
        if normal_clip.shape != incident_clip.shape:
            raise ValueError(
                "normal_clip and incident_clip must have identical shapes; "
                f"got {normal_clip.shape} vs {incident_clip.shape}."
            )
        _t, h, w, _c = normal_clip.shape
        out = normal_clip.copy()

        # Extract incident crop region.
        if incident_bbox is not None:
            ix1 = int(incident_bbox[0] * w)
            iy1 = int(incident_bbox[1] * h)
            ix2 = int(incident_bbox[2] * w)
            iy2 = int(incident_bbox[3] * h)
        else:
            ix1, iy1, ix2, iy2 = 0, 0, w, h

        # Determine paste size.
        min_side = min(h, w)
        patch_size = int(
            self._rng.uniform(*resize_range) * min_side
        )
        patch_h = patch_size
        patch_w = int(patch_size * max(1, (ix2 - ix1)) / max(1, (iy2 - iy1)))

        # Determine paste location.
        if target_region is not None:
            tx1 = int(target_region[0] * w)
            ty1 = int(target_region[1] * h)
            tx2 = min(w, tx1 + patch_w)
            ty2 = min(h, ty1 + patch_h)
        else:
            tx1 = int(self._rng.integers(0, max(1, w - patch_w)))
            ty1 = int(self._rng.integers(0, max(1, h - patch_h)))
            tx2 = min(w, tx1 + patch_w)
            ty2 = min(h, ty1 + patch_h)

        paste_h = ty2 - ty1
        paste_w = tx2 - tx1
        if paste_h <= 0 or paste_w <= 0:
            return out

        for t in range(_t):
            crop = incident_clip[t, iy1:iy2, ix1:ix2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            resized = cv2.resize(
                crop, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR
            )
            bg = out[t, ty1:ty2, tx1:tx2]
            out[t, ty1:ty2, tx1:tx2] = np.clip(
                blend_alpha * resized + (1.0 - blend_alpha) * bg, 0.0, 1.0
            )

        return out

    # ------------------------------------------------------------------
    # Compose multiple augmentations
    # ------------------------------------------------------------------

    def compose(
        self,
        clip: np.ndarray,
        transforms: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Apply a sequential list of named augmentations.

        Each entry in ``transforms`` is a dict with ``"name"`` (str) and
        optional ``"kwargs"`` (dict).  Example::

            [
                {"name": "photometric_distortion", "kwargs": {"brightness_delta": 0.2}},
                {"name": "synthetic_fog", "kwargs": {"fog_density": 0.3}},
                {"name": "cutout"},
            ]

        Returns
        -------
        np.ndarray
            Augmented clip.
        """
        result = clip
        for spec in transforms:
            name = spec["name"]
            kwargs = spec.get("kwargs", {})
            fn = getattr(self, name, None)
            if fn is None:
                raise ValueError(f"Unknown augmentation: '{name}'.")
            result = fn(result, **kwargs)
        return result

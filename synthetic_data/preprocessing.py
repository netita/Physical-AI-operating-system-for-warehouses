"""
warehousegpt.synthetic_data.preprocessing
==========================================
Frame-level and temporal preprocessing transforms for warehouse video data.

All public methods operate on ``numpy`` arrays and return ``numpy`` arrays so
they remain independent of the training framework.  Callers that need
``torch.Tensor`` output should apply ``torch.from_numpy`` afterwards.

Coordinate conventions
-----------------------
* Frames / RGB arrays: ``[H, W, C]`` uint8 or float32 in ``[0, 1]``.
* Depth maps:          ``[H, W]``    float32, metres.
* Clips:               ``[T, H, W, C]`` (time-first).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet channel-wise statistics (RGB order).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class VideoPreprocessor:
    """
    Stateless collection of video / depth preprocessing transforms.

    All methods are deliberately stateless (classmethod or staticmethod) so
    they can be serialised and used inside worker processes without pickling
    state.

    Parameters
    ----------
    mean:
        Per-channel normalisation mean (RGB).  Defaults to ImageNet stats.
    std:
        Per-channel normalisation std (RGB).  Defaults to ImageNet stats.
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    # ------------------------------------------------------------------
    # Frame normalisation
    # ------------------------------------------------------------------

    def normalize_frames(
        self,
        frames: np.ndarray,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Normalise video frames to zero-mean, unit-variance per channel.

        Parameters
        ----------
        frames:
            ``[T, H, W, C]`` uint8 (0–255) or float32 (0.0–1.0) array.
        mean:
            Override per-channel mean.  Defaults to instance mean.
        std:
            Override per-channel std.  Defaults to instance std.

        Returns
        -------
        np.ndarray
            ``[T, H, W, C]`` float32 normalised frames.
        """
        mu = (
            np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
            if mean is not None
            else self.mean[np.newaxis]
        )
        sigma = (
            np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
            if std is not None
            else self.std[np.newaxis]
        )

        out = frames.astype(np.float32)
        if out.max() > 1.5:  # heuristic: assume uint8 range
            out /= 255.0
        out = (out - mu) / (sigma + 1e-7)
        return out

    def denormalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Invert ``normalize_frames`` and clip to [0, 1]."""
        out = frames * (self.std[np.newaxis] + 1e-7) + self.mean[np.newaxis]
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Spatial resize
    # ------------------------------------------------------------------

    @staticmethod
    def resize_preserve_aspect(
        frames: np.ndarray,
        target_h: int,
        target_w: int,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        Resize frames to fit within ``(target_h, target_w)`` while preserving
        the original aspect ratio.  The result is padded with black pixels to
        reach the exact target size (letter/pillar-boxing).

        Parameters
        ----------
        frames:
            ``[T, H, W, C]`` or ``[H, W, C]`` array.
        target_h:
            Maximum output height in pixels.
        target_w:
            Maximum output width in pixels.
        interpolation:
            OpenCV interpolation flag.

        Returns
        -------
        np.ndarray
            Padded frames of shape ``[T, target_h, target_w, C]`` or
            ``[target_h, target_w, C]``.
        """
        single = frames.ndim == 3
        if single:
            frames = frames[np.newaxis]

        _t, src_h, src_w, c = frames.shape
        scale = min(target_h / src_h, target_w / src_w)
        new_h = int(round(src_h * scale))
        new_w = int(round(src_w * scale))

        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        dtype = frames.dtype
        out = np.zeros((_t, target_h, target_w, c), dtype=dtype)
        for t_idx, frame in enumerate(frames):
            resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
            if resized.ndim == 2:  # grayscale safety
                resized = resized[:, :, np.newaxis]
            out[t_idx, pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        return out[0] if single else out

    # ------------------------------------------------------------------
    # Depth alignment
    # ------------------------------------------------------------------

    @staticmethod
    def align_depth_to_rgb(
        depth: np.ndarray,
        calibration: dict[str, Any],
        rgb_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Project a depth map captured by a separate depth sensor into the
        reference frame of the RGB colour camera.

        The transformation applies the rigid-body extrinsic (R, t) between
        the two sensors, reprojects each depth pixel into 3-D space, then
        projects it into the colour image plane using the colour intrinsics.

        Parameters
        ----------
        depth:
            ``[Hd, Wd]`` float32 depth image in metres (depth-camera frame).
        calibration:
            Dictionary with keys:
                ``fx``, ``fy``, ``cx``, ``cy``  – colour camera intrinsics
                ``extrinsic_R``                   – [3,3] rotation D→RGB
                ``extrinsic_t``                   – [3]   translation D→RGB
                ``depth_fx``, ``depth_fy``,
                ``depth_cx``, ``depth_cy``        – depth camera intrinsics
                                                    (optional; falls back to
                                                    colour intrinsics when
                                                    absent)
        rgb_hw:
            Output resolution ``(H, W)`` for the colour image plane.  When
            ``None``, the depth image resolution is used.

        Returns
        -------
        np.ndarray
            ``[H_rgb, W_rgb]`` float32 aligned depth map in metres.  Pixels
            that no depth point projects to are filled with ``0.0``.
        """
        depth = depth.astype(np.float32)
        hd, wd = depth.shape
        out_h, out_w = rgb_hw if rgb_hw is not None else (hd, wd)

        # Colour camera intrinsics.
        c_fx: float = float(calibration.get("fx", 910.0))
        c_fy: float = float(calibration.get("fy", 910.0))
        c_cx: float = float(calibration.get("cx", out_w / 2.0))
        c_cy: float = float(calibration.get("cy", out_h / 2.0))

        # Depth camera intrinsics (may equal colour if sensors share optics).
        d_fx: float = float(calibration.get("depth_fx", c_fx))
        d_fy: float = float(calibration.get("depth_fy", c_fy))
        d_cx: float = float(calibration.get("depth_cx", wd / 2.0))
        d_cy: float = float(calibration.get("depth_cy", hd / 2.0))

        R: np.ndarray = np.asarray(
            calibration.get("extrinsic_R", np.eye(3)), dtype=np.float64
        ).reshape(3, 3)
        t: np.ndarray = np.asarray(
            calibration.get("extrinsic_t", np.zeros(3)), dtype=np.float64
        ).ravel()

        # Build pixel grid in depth-camera space.
        v_idx, u_idx = np.indices((hd, wd), dtype=np.float32)
        z = depth  # [Hd, Wd]
        valid = z > 0.0

        # Back-project to 3-D (depth camera frame).
        x_d = (u_idx - d_cx) * z / d_fx
        y_d = (v_idx - d_cy) * z / d_fy

        # Stack into [N, 3] for valid pixels only.
        pts_d = np.stack([x_d[valid], y_d[valid], z[valid]], axis=1)  # [N, 3]

        # Apply extrinsic: pts_rgb = R @ pts_d.T + t
        pts_rgb = (R @ pts_d.T + t[:, np.newaxis]).T  # [N, 3]

        # Project into colour image plane.
        z_rgb = pts_rgb[:, 2]
        pos_mask = z_rgb > 0.0
        u_c = (c_fx * pts_rgb[pos_mask, 0] / z_rgb[pos_mask] + c_cx).round().astype(int)
        v_c = (c_fy * pts_rgb[pos_mask, 1] / z_rgb[pos_mask] + c_cy).round().astype(int)
        z_c = z_rgb[pos_mask]

        # Keep only pixels within the output image bounds.
        in_bounds = (u_c >= 0) & (u_c < out_w) & (v_c >= 0) & (v_c < out_h)
        u_c, v_c, z_c = u_c[in_bounds], v_c[in_bounds], z_c[in_bounds]

        aligned = np.zeros((out_h, out_w), dtype=np.float32)
        # Use minimum depth for overlapping projections (nearest-surface wins).
        np.minimum.at(aligned, (v_c, u_c), z_c.astype(np.float32))

        return aligned

    # ------------------------------------------------------------------
    # Temporal clip sampling
    # ------------------------------------------------------------------

    @staticmethod
    def temporal_clip_sampling(
        video: np.ndarray,
        clip_len: int,
        stride: int = 1,
        pad_last: bool = True,
    ) -> list[np.ndarray]:
        """
        Slice a video into fixed-length temporal clips with optional stride.

        Parameters
        ----------
        video:
            ``[T, H, W, C]`` video array.
        clip_len:
            Number of frames per clip.
        stride:
            Step between consecutive clip start frames.
        pad_last:
            If ``True``, the final clip is zero-padded to ``clip_len`` when
            the video length is not a multiple of ``stride * clip_len``.

        Returns
        -------
        list[np.ndarray]
            List of ``[clip_len, H, W, C]`` clips.
        """
        if video.ndim != 4:
            raise ValueError(
                f"Expected video of shape [T, H, W, C], got shape {video.shape}."
            )
        total_frames, h, w, c = video.shape
        if clip_len > total_frames:
            if pad_last:
                pad = np.zeros(
                    (clip_len - total_frames, h, w, c), dtype=video.dtype
                )
                return [np.concatenate([video, pad], axis=0)]
            return [video] if total_frames > 0 else []

        clips: list[np.ndarray] = []
        for start in range(0, total_frames - clip_len + 1, stride):
            clips.append(video[start : start + clip_len])

        # Handle tail.
        last_start = clips[-1].shape[0] if clips else 0
        covered = (len(clips) - 1) * stride + clip_len if clips else 0
        if pad_last and covered < total_frames:
            tail = video[covered:]
            if tail.shape[0] > 0:
                pad = np.zeros(
                    (clip_len - tail.shape[0], h, w, c), dtype=video.dtype
                )
                clips.append(np.concatenate([tail, pad], axis=0))

        _ = last_start  # suppress unused warning
        return clips

    # ------------------------------------------------------------------
    # Convenience: apply full spatial pipeline to a clip
    # ------------------------------------------------------------------

    def preprocess_clip(
        self,
        clip: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """
        Resize + normalise a ``[T, H, W, C]`` clip in one call.

        Returns a ``[T, target_h, target_w, C]`` float32 array.
        """
        resized = self.resize_preserve_aspect(clip, target_h, target_w)
        return self.normalize_frames(resized)


def compute_optical_flow(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
) -> np.ndarray:
    """
    Compute dense Farneback optical flow between two consecutive frames.

    Parameters
    ----------
    prev_frame / next_frame:
        ``[H, W, C]`` uint8 or float32 RGB frames.

    Returns
    -------
    np.ndarray
        ``[H, W, 2]`` float32 flow field (dx, dy) in pixels.
    """

    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    prev_gray = _to_gray(prev_frame)
    next_gray = _to_gray(next_frame)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return flow  # [H, W, 2]

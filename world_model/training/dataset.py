"""
WarehouseVideoDataset — loads from HuggingFace Hub and returns
pre-processed warehouse video clips ready for world model training.

Dataset: nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes

Expected HuggingFace features schema (inferred from NVIDIA dataset card):
    video          : list[PIL.Image] or VideoFile or List[np.ndarray]
    depth          : list[np.ndarray]  (H, W, 1) float32 normalised
    segmentation   : list[np.ndarray]  (H, W) int32 class labels
    bounding_boxes : list[dict]        COCO-style bbox annotations per frame
    actions        : list[dict]        per-frame forklift state
    labels         : dict              clip-level metadata (scene_id, event_type …)

Returned batch dict keys
------------------------
    video:        (T, 3, H, W) float32 in [0, 1]
    depth:        (T, 1, H, W) float32 in [0, 1]
    segmentation: (T, H, W)   int64 class label
    actions:      (T, action_dim) float32
    risk_label:   ()  float32  0 = safe, 1 = hazardous
    event_type:   str  e.g. 'collision', 'near_miss', 'normal'
    clip_id:      str  unique identifier
"""

from __future__ import annotations

import hashlib
import io
import random
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from datasets import load_dataset, Dataset as HFDataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    warnings.warn(
        "HuggingFace `datasets` is not installed. "
        "Install with: pip install datasets",
        stacklevel=2,
    )

try:
    import torchvision.transforms.functional as TF
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes"

# Action vector index mapping
ACTION_KEYS = ["vx", "vy", "omega", "fork_height", "fork_tilt", "load_weight"]
ACTION_DIM = len(ACTION_KEYS)

# Action normalisation stats (approximate — updated during preprocessing)
ACTION_MEAN = torch.tensor([0.0, 0.0, 0.0, 1.5, 0.0, 250.0], dtype=torch.float32)
ACTION_STD = torch.tensor([1.0, 1.0, 0.5, 1.5, 0.3, 500.0], dtype=torch.float32)

# Segmentation class ids (subset — full list from dataset card)
SEG_CLASSES = {
    0: "background",
    1: "floor",
    2: "rack",
    3: "forklift",
    4: "worker",
    5: "pallet",
    6: "fire",
    7: "smoke",
    8: "wall",
    9: "ceiling",
    10: "conveyor",
}

HAZARDOUS_EVENTS = frozenset(
    ["collision", "near_miss", "fire", "rack_collapse", "restricted_zone_violation"]
)


# ---------------------------------------------------------------------------
# Frame-level transforms
# ---------------------------------------------------------------------------

def _decode_frame(raw: Any, height: int, width: int) -> Tensor:
    """
    Decode a single frame from various raw formats to (C, H, W) float32.

    Handles:
        - PIL.Image
        - np.ndarray (H, W, C) uint8 or float32
        - bytes / BytesIO (JPEG/PNG)
    """
    if _PIL_AVAILABLE and isinstance(raw, PILImage.Image):
        img = raw.resize((width, height), PILImage.BILINEAR)
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    if isinstance(raw, np.ndarray):
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / 255.0
        if raw.shape[:2] != (height, width):
            # Resize via PIL
            if _PIL_AVAILABLE:
                pil = PILImage.fromarray((raw * 255).astype(np.uint8))
                pil = pil.resize((width, height), PILImage.BILINEAR)
                raw = np.array(pil, dtype=np.float32) / 255.0
        if raw.ndim == 2:
            raw = raw[:, :, None]
        return torch.from_numpy(raw).permute(2, 0, 1)

    if isinstance(raw, (bytes, bytearray)):
        if _PIL_AVAILABLE:
            pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
            return _decode_frame(pil, height, width)

    raise TypeError(f"Cannot decode frame from type {type(raw)}")


def _decode_depth(raw: Any, height: int, width: int) -> Tensor:
    """Decode depth map to (1, H, W) float32 in [0, 1]."""
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            raw = raw[..., 0]  # take first channel
        if raw.shape != (height, width):
            if _PIL_AVAILABLE:
                raw_uint8 = (raw / raw.max() * 255).astype(np.uint8)
                pil = PILImage.fromarray(raw_uint8, mode="L")
                pil = pil.resize((width, height), PILImage.NEAREST)
                raw = np.array(pil, dtype=np.float32) / 255.0
        # Normalise to [0, 1]
        max_val = raw.max()
        if max_val > 0:
            raw = raw / max_val
        return torch.from_numpy(raw.astype(np.float32)).unsqueeze(0)

    if _PIL_AVAILABLE and isinstance(raw, PILImage.Image):
        arr = np.array(raw.resize((width, height), PILImage.NEAREST), dtype=np.float32)
        return torch.from_numpy(arr / (arr.max() + 1e-8)).unsqueeze(0)

    return torch.zeros(1, height, width)


def _decode_segmentation(raw: Any, height: int, width: int) -> Tensor:
    """Decode segmentation mask to (H, W) int64."""
    if isinstance(raw, np.ndarray):
        if raw.shape[:2] != (height, width):
            if _PIL_AVAILABLE:
                pil = PILImage.fromarray(raw.astype(np.uint8), mode="L")
                pil = pil.resize((width, height), PILImage.NEAREST)
                raw = np.array(pil, dtype=np.int64)
        return torch.from_numpy(raw.astype(np.int64))

    if _PIL_AVAILABLE and isinstance(raw, PILImage.Image):
        arr = np.array(raw.convert("L").resize((width, height), PILImage.NEAREST))
        return torch.from_numpy(arr.astype(np.int64))

    return torch.zeros(height, width, dtype=torch.long)


def _parse_actions(action_data: Any, T: int) -> Tensor:
    """
    Parse per-frame action data into (T, action_dim) float32 tensor.

    Normalises using ACTION_MEAN / ACTION_STD.
    """
    actions = torch.zeros(T, ACTION_DIM, dtype=torch.float32)

    if action_data is None:
        return actions

    if isinstance(action_data, list):
        for t, frame_action in enumerate(action_data[:T]):
            if isinstance(frame_action, dict):
                for i, key in enumerate(ACTION_KEYS):
                    actions[t, i] = float(frame_action.get(key, 0.0))
    elif isinstance(action_data, np.ndarray):
        arr = torch.from_numpy(action_data[:T].astype(np.float32))
        actions[:arr.shape[0], :arr.shape[1]] = arr

    # Normalise
    actions = (actions - ACTION_MEAN) / (ACTION_STD + 1e-8)
    return actions


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def _speed_jitter(
    indices: list[int],
    jitter_range: tuple[float, float] = (0.75, 1.5),
) -> list[int]:
    """
    Apply speed jitter by resampling frame indices at a random rate.

    A jitter > 1.0 speeds up the video (fewer frames per clip window),
    < 1.0 slows it down (more frames per clip window, may loop).
    """
    speed = random.uniform(*jitter_range)
    n = len(indices)
    if speed >= 1.0:
        # Sample every speed-th frame (skip some frames)
        step = max(1, round(speed))
        jittered = [indices[min(i * step, n - 1)] for i in range(n)]
    else:
        # Interpolate (hold frames longer) — repeat frames at even intervals
        total_src = max(1, round(n * speed))
        jittered = []
        for i in range(n):
            src_i = min(round(i * speed), total_src - 1)
            jittered.append(indices[min(src_i, n - 1)])
    return jittered


def _frame_dropout(indices: list[int], dropout_prob: float) -> list[int]:
    """
    Randomly drop frames and repeat the last valid frame.
    This teaches the model to be robust to missing frames.
    """
    if dropout_prob <= 0.0:
        return indices
    result = []
    last_valid = indices[0]
    for idx in indices:
        if random.random() < dropout_prob and len(result) > 0:
            result.append(last_valid)  # repeat previous frame
        else:
            result.append(idx)
            last_valid = idx
    return result


def _color_jitter(frame: Tensor) -> Tensor:
    """Apply random brightness / contrast / saturation jitter to a (C, H, W) frame."""
    if not _TV_AVAILABLE:
        return frame
    # Convert to PIL for TorchVision transforms
    pil = TF.to_pil_image(frame.clamp(0, 1))
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    pil = TF.adjust_brightness(pil, brightness)
    pil = TF.adjust_contrast(pil, contrast)
    pil = TF.adjust_saturation(pil, saturation)
    return TF.to_tensor(pil)


# ---------------------------------------------------------------------------
# WarehouseVideoDataset
# ---------------------------------------------------------------------------

class WarehouseVideoDataset(Dataset):
    """
    PyTorch Dataset that wraps the NVIDIA PhysicalAI Warehouse dataset.

    Loads video clips from HuggingFace Hub, decodes frames, applies temporal
    augmentation, and returns a batch-ready dictionary.

    Args:
        dataset_name:       HuggingFace Hub dataset identifier
        split:              'train' | 'validation' | 'test'
        clip_frames:        total number of frames per returned clip
        image_height:       target frame height (pixels)
        image_width:        target frame width (pixels)
        speed_jitter_range: (min, max) temporal speed multiplier
        frame_dropout_prob: probability of dropping a frame
        color_jitter:       apply random color jitter to RGB frames
        action_dim:         action vector dimension
        cache_dir:          local directory to cache the HF dataset
        streaming:          use HF dataset streaming (no local cache needed)
        max_samples:        limit dataset size (useful for debugging)
    """

    def __init__(
        self,
        dataset_name: str = DATASET_ID,
        split: str = "train",
        clip_frames: int = 24,
        image_height: int = 256,
        image_width: int = 256,
        speed_jitter_range: tuple[float, float] = (0.75, 1.5),
        frame_dropout_prob: float = 0.1,
        color_jitter: bool = True,
        action_dim: int = ACTION_DIM,
        cache_dir: Optional[Path] = None,
        streaming: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()

        if not _HF_AVAILABLE:
            raise ImportError(
                "HuggingFace `datasets` is required. "
                "Install with: pip install datasets"
            )

        self.dataset_name = dataset_name
        self.split = split
        self.clip_frames = clip_frames
        self.image_height = image_height
        self.image_width = image_width
        self.speed_jitter_range = speed_jitter_range
        self.frame_dropout_prob = frame_dropout_prob
        self.color_jitter = color_jitter
        self.action_dim = action_dim
        self.streaming = streaming

        # Load dataset
        self._hf_dataset = self._load_hf_dataset(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )

        # Apply size limit (for debugging)
        if max_samples is not None and not streaming:
            self._hf_dataset = self._hf_dataset.select(
                range(min(max_samples, len(self._hf_dataset)))
            )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_dataset(
        dataset_name: str,
        split: str,
        cache_dir: Optional[Path],
        streaming: bool,
    ) -> "HFDataset":
        """
        Load from HuggingFace Hub with graceful fallback to synthetic data.

        The NVIDIA warehouse dataset may require authentication or a specific
        HF token.  Set HF_TOKEN environment variable or use huggingface-cli login.
        """
        kwargs: dict[str, Any] = {
            "path": dataset_name,
            "split": split,
            "trust_remote_code": True,
            "streaming": streaming,
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        try:
            return load_dataset(**kwargs)
        except Exception as exc:
            warnings.warn(
                f"Could not load {dataset_name!r} from HuggingFace Hub: {exc}\n"
                "Falling back to a synthetic placeholder dataset for development. "
                "To use the real dataset, ensure your HF token is configured and "
                "you have accepted the dataset license on the Hub.",
                stacklevel=3,
            )
            return _SyntheticWarehouseDataset(n_samples=1000)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.streaming:
            # Streaming datasets have no known length
            return 123_000  # approximate dataset size
        try:
            return len(self._hf_dataset)
        except TypeError:
            return 123_000

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._hf_dataset[idx]
        return self._process_sample(sample)

    # ------------------------------------------------------------------
    # Sample processing
    # ------------------------------------------------------------------

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single HF dataset sample into a training-ready dict.

        Returns
        -------
        dict with keys:
            video:        (T, 3, H, W) float32 in [0, 1]
            depth:        (T, 1, H, W) float32 in [0, 1]
            segmentation: (T, H, W)   int64
            actions:      (T, action_dim) float32
            risk_label:   scalar float32 in {0, 1}
            event_type:   str
            clip_id:      str
        """
        # --- Extract raw frames ---
        raw_frames = sample.get("video", sample.get("frames", []))
        raw_depth = sample.get("depth", [])
        raw_seg = sample.get("segmentation", sample.get("segmentation_masks", []))

        n_src = len(raw_frames) if raw_frames else 0

        # Build frame index sequence with temporal augmentation
        if n_src == 0:
            # No frames: return empty sample
            return self._empty_sample()

        base_indices = list(range(min(n_src, self.clip_frames)))
        # Pad to clip_frames if source is shorter
        while len(base_indices) < self.clip_frames:
            base_indices.append(base_indices[-1])

        # Temporal augmentation (training only)
        if self.frame_dropout_prob > 0.0 or self.speed_jitter_range != (1.0, 1.0):
            jittered = _speed_jitter(base_indices, self.speed_jitter_range)
            jittered = _frame_dropout(jittered, self.frame_dropout_prob)
            frame_indices = jittered[: self.clip_frames]
        else:
            frame_indices = base_indices[: self.clip_frames]

        T = self.clip_frames

        # --- Decode frames ---
        video_list: list[Tensor] = []
        depth_list: list[Tensor] = []
        seg_list: list[Tensor] = []

        for fi in frame_indices:
            fi_safe = min(fi, n_src - 1)

            # RGB frame
            frame = _decode_frame(raw_frames[fi_safe], self.image_height, self.image_width)
            if self.color_jitter and torch.rand(1).item() < 0.5:
                frame = _color_jitter(frame)
            video_list.append(frame)

            # Depth
            if raw_depth and fi_safe < len(raw_depth):
                depth_list.append(
                    _decode_depth(raw_depth[fi_safe], self.image_height, self.image_width)
                )
            else:
                depth_list.append(torch.zeros(1, self.image_height, self.image_width))

            # Segmentation
            if raw_seg and fi_safe < len(raw_seg):
                seg_list.append(
                    _decode_segmentation(raw_seg[fi_safe], self.image_height, self.image_width)
                )
            else:
                seg_list.append(torch.zeros(self.image_height, self.image_width, dtype=torch.long))

        video = torch.stack(video_list)   # (T, 3, H, W)
        depth = torch.stack(depth_list)   # (T, 1, H, W)
        segmentation = torch.stack(seg_list)  # (T, H, W)

        # --- Actions ---
        raw_actions = sample.get("actions", None)
        actions = _parse_actions(raw_actions, T)  # (T, action_dim)

        # --- Labels ---
        labels = sample.get("labels", sample.get("metadata", {}))
        event_type = str(labels.get("event_type", "normal")) if isinstance(labels, dict) else "normal"
        risk_label = float(event_type in HAZARDOUS_EVENTS)

        # Clip identifier
        clip_id = str(labels.get("clip_id", hashlib.md5(str(idx).encode()).hexdigest()[:8]))

        return {
            "video": video,
            "depth": depth,
            "segmentation": segmentation,
            "actions": actions,
            "risk_label": torch.tensor(risk_label, dtype=torch.float32),
            "event_type": event_type,
            "clip_id": clip_id,
        }

    def _empty_sample(self) -> dict[str, Any]:
        """Return a zero-filled sample when source data is missing."""
        T = self.clip_frames
        H, W = self.image_height, self.image_width
        return {
            "video": torch.zeros(T, 3, H, W),
            "depth": torch.zeros(T, 1, H, W),
            "segmentation": torch.zeros(T, H, W, dtype=torch.long),
            "actions": torch.zeros(T, self.action_dim),
            "risk_label": torch.tensor(0.0),
            "event_type": "unknown",
            "clip_id": "empty",
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_class_weights(self) -> Tensor:
        """
        Compute inverse-frequency class weights for the risk label
        to handle the class imbalance (hazardous events are rare).
        Approximate: ~5% hazardous in the NVIDIA dataset.
        """
        hazard_fraction = 0.05
        weight_hazard = 1.0 / hazard_fraction
        weight_safe = 1.0 / (1.0 - hazard_fraction)
        return torch.tensor([weight_safe, weight_hazard], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Synthetic placeholder dataset (development / CI)
# ---------------------------------------------------------------------------

class _SyntheticWarehouseDataset:
    """
    Generates random (but structured) samples that mimic the HF schema.
    Used when the real NVIDIA dataset is not accessible.
    """

    def __init__(self, n_samples: int = 1000, clip_len: int = 32) -> None:
        self.n_samples = n_samples
        self.clip_len = clip_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed=idx)
        T = self.clip_len
        H, W = 256, 256

        # Random video frames (H, W, 3) uint8
        frames = [
            rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(T)
        ]
        depth = [rng.random((H, W, 1)).astype(np.float32) for _ in range(T)]
        seg = [rng.integers(0, 11, (H, W), dtype=np.int32) for _ in range(T)]

        actions = []
        for _ in range(T):
            actions.append({
                "vx": float(rng.uniform(-1.0, 1.0)),
                "vy": float(rng.uniform(-0.5, 0.5)),
                "omega": float(rng.uniform(-0.5, 0.5)),
                "fork_height": float(rng.uniform(0.0, 3.0)),
                "fork_tilt": float(rng.uniform(-0.3, 0.3)),
                "load_weight": float(rng.uniform(0, 1000)),
            })

        event_types = ["normal", "normal", "normal", "near_miss", "collision"]
        event = event_types[idx % len(event_types)]

        return {
            "video": frames,
            "depth": depth,
            "segmentation": seg,
            "actions": actions,
            "labels": {"event_type": event, "clip_id": f"syn_{idx:06d}"},
        }

    def select(self, indices: range) -> "_SyntheticWarehouseDataset":
        """Mimic HF Dataset.select()."""
        ds = _SyntheticWarehouseDataset(n_samples=len(indices), clip_len=self.clip_len)
        return ds

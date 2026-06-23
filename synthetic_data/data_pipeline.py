"""
warehousegpt.synthetic_data.data_pipeline
==========================================
End-to-end DataLoader construction for multi-modal warehouse video data.

The pipeline handles:
* Video frames (RGB clips)         – ``[T, H, W, C]`` float32
* Depth maps                       – ``[T, H, W]``    float32, metres
* Segmentation masks               – ``[T, H, W]``    int32
* Camera calibration per sample    – dict of numpy arrays / scalars
* Bounding boxes + class labels    – padded tensors

Architecture
------------
``DataPipeline`` is the central class.  It wraps a ``WarehouseHFDataset``
(or any mapping/sequence that returns the same schema) and applies
``VideoPreprocessor`` + ``WarehouseAugmentation`` inside a
``torch.utils.data.Dataset`` adapter, then hands off to
``torch.utils.data.DataLoader``.

Optional Ray Data backend
~~~~~~~~~~~~~~~~~~~~~~~~~
When ``use_ray=True`` the pipeline uses ``ray.data`` for distributed
pre-processing.  Ray is an optional dependency; the pipeline falls back
gracefully to the standard PyTorch DataLoader when Ray is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Scenario label → integer class index mapping.
SCENARIO_TO_IDX: dict[str, int] = {
    "normal": 0,
    "near_miss": 1,
    "collision": 2,
    "fire": 3,
}


@dataclass
class PipelineConfig:
    """Hyper-parameters controlling clip extraction and batching."""

    # Spatial resolution fed to the model.
    frame_height: int = 224
    frame_width: int = 224
    # Temporal clip length (frames).
    clip_len: int = 16
    # Stride between consecutive clip start frames during training.
    clip_stride: int = 8
    # ImageNet normalisation statistics.
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Maximum bounding boxes per sample (pad / truncate to this).
    max_bboxes: int = 64
    # Random seed for augmentation reproducibility.
    aug_seed: int = 42


def _collate_warehouse(
    batch: list[dict[str, Any]],
    max_bboxes: int = 64,
) -> dict[str, Any]:
    """
    Custom collate function for multi-modal warehouse samples.

    Stacks tensors where possible, pads bounding boxes to a fixed size,
    and keeps calibration as a list of dicts.
    """
    import torch  # noqa: PLC0415

    def _stack(key: str) -> "torch.Tensor | None":
        vals = [item[key] for item in batch if item.get(key) is not None]
        if not vals:
            return None
        arrs = [np.asarray(v, dtype=np.float32) for v in vals]
        return torch.from_numpy(np.stack(arrs, axis=0))

    # RGB clip: [B, T, H, W, C].
    video = _stack("video")
    # Depth: [B, T, H, W].
    depth = _stack("depth")
    # Segmentation: [B, T, H, W].
    seg = _stack("segmentation")

    # Bounding boxes: pad to [B, max_bboxes, 4] and labels to [B, max_bboxes].
    all_boxes = []
    all_labels = []
    for item in batch:
        bboxes = np.asarray(item.get("bboxes") or [], dtype=np.float32)
        labels = np.asarray(item.get("bbox_labels") or [], dtype=np.int64)
        n = min(len(bboxes), max_bboxes)
        box_pad = np.zeros((max_bboxes, 4), dtype=np.float32)
        lbl_pad = np.full((max_bboxes,), -1, dtype=np.int64)
        if n > 0:
            box_pad[:n] = bboxes[:n]
            lbl_pad[:n] = labels[:n]
        all_boxes.append(box_pad)
        all_labels.append(lbl_pad)

    bboxes_t = torch.from_numpy(np.stack(all_boxes, axis=0))
    labels_t = torch.from_numpy(np.stack(all_labels, axis=0))

    # Scenario class index.
    scenario_indices = torch.tensor(
        [item.get("scenario_idx", 0) for item in batch], dtype=torch.long
    )

    return {
        "video": video,               # [B, T, H, W, C] or None
        "depth": depth,               # [B, T, H, W] or None
        "segmentation": seg,          # [B, T, H, W] or None
        "bboxes": bboxes_t,           # [B, max_bboxes, 4]
        "bbox_labels": labels_t,      # [B, max_bboxes]
        "scenario_idx": scenario_indices,  # [B]
        "sample_ids": [item["sample_id"] for item in batch],
        "calibration": [item.get("calibration") for item in batch],
    }


class _WarehouseTorchDataset:
    """
    Adapts a ``WarehouseHFDataset`` (or any length/getitem mapping) into a
    ``torch.utils.data.Dataset`` that applies preprocessing and augmentation.

    This class deliberately avoids inheriting from ``torch.utils.data.Dataset``
    at class-definition time so the module can be imported without PyTorch.
    At construction time, if PyTorch is available, the MRO is patched.
    """

    def __init__(
        self,
        source: Any,
        preprocessor: "Any",  # VideoPreprocessor
        config: PipelineConfig,
        augmentation: "Any | None" = None,   # WarehouseAugmentation | None
    ) -> None:
        # Attempt to inherit torch Dataset protocol at runtime.
        try:
            import torch.utils.data  # noqa: PLC0415

            if not isinstance(self, torch.utils.data.Dataset):
                self.__class__ = type(
                    "_WarehouseTorchDatasetPatched",
                    (torch.utils.data.Dataset, _WarehouseTorchDataset),
                    {},
                )
        except ImportError:
            pass

        self._source = source
        self._preprocessor = preprocessor
        self._config = config
        self._aug = augmentation

        # Pre-expand clips from the source dataset into flat (sample_idx, clip_idx)
        # index pairs so __len__ / __getitem__ are O(1).
        self._index: list[tuple[int, int]] = []
        self._build_clip_index()

    def _build_clip_index(self) -> None:
        """Iterate source samples and expand each video into clip positions."""
        cfg = self._config
        for sample_idx in range(len(self._source)):
            raw = self._source[sample_idx]
            video = raw.get("video") or raw.get("rgb")
            if video is None:
                # No video: single "clip" placeholder.
                self._index.append((sample_idx, 0))
                continue

            vid_arr = np.asarray(video, dtype=np.float32)
            if vid_arr.ndim == 3:
                # Single frame.
                n_clips = 1
            else:
                t = vid_arr.shape[0]
                n_clips = max(
                    1,
                    len(
                        range(0, max(1, t - cfg.clip_len + 1), cfg.clip_stride)
                    ),
                )
            for clip_idx in range(n_clips):
                self._index.append((sample_idx, clip_idx))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_idx, clip_idx = self._index[idx]
        raw = self._source[sample_idx]
        cfg = self._config

        # ----- Video -----
        video_raw = raw.get("video") or raw.get("rgb")
        if video_raw is not None:
            vid_arr = np.asarray(video_raw, dtype=np.float32)
            if vid_arr.ndim == 3:
                vid_arr = vid_arr[np.newaxis]
            start = clip_idx * cfg.clip_stride
            clip = vid_arr[start : start + cfg.clip_len]
            # Pad if needed.
            if clip.shape[0] < cfg.clip_len:
                pad = np.zeros(
                    (cfg.clip_len - clip.shape[0], *clip.shape[1:]), dtype=np.float32
                )
                clip = np.concatenate([clip, pad], axis=0)
            # Preprocess.
            clip = self._preprocessor.resize_preserve_aspect(
                clip, cfg.frame_height, cfg.frame_width
            )
            clip = self._preprocessor.normalize_frames(
                clip, mean=cfg.norm_mean, std=cfg.norm_std
            )
            # Augment (train only – augmentation is None during eval).
            if self._aug is not None:
                clip = self._aug.photometric_distortion(clip, p=0.5)
                clip = self._aug.cutout(clip, n_holes=2, p=0.3)
        else:
            clip = None

        # ----- Depth -----
        depth_raw = raw.get("depth")
        depth_clips: np.ndarray | None = None
        if depth_raw is not None:
            # depth_raw may be a single [H,W] map or a [T,H,W] sequence.
            darr = np.asarray(depth_raw, dtype=np.float32)
            if darr.ndim == 2:
                darr = np.stack([darr] * cfg.clip_len, axis=0)
            start = clip_idx * cfg.clip_stride
            d_clip = darr[start : start + cfg.clip_len]
            if d_clip.shape[0] < cfg.clip_len:
                pad = np.zeros(
                    (cfg.clip_len - d_clip.shape[0], *d_clip.shape[1:]), dtype=np.float32
                )
                d_clip = np.concatenate([d_clip, pad], axis=0)
            # Resize depth maps independently.
            resized_d = []
            for d_frame in d_clip:
                import cv2  # noqa: PLC0415

                r = cv2.resize(
                    d_frame, (cfg.frame_width, cfg.frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                resized_d.append(r)
            depth_clips = np.stack(resized_d, axis=0)  # [T, H, W]

        # ----- Segmentation -----
        seg_raw = raw.get("segmentation")
        seg_clips: np.ndarray | None = None
        if seg_raw is not None:
            sarr = np.asarray(seg_raw, dtype=np.int32)
            if sarr.ndim == 2:
                sarr = np.stack([sarr] * cfg.clip_len, axis=0)
            start = clip_idx * cfg.clip_stride
            s_clip = sarr[start : start + cfg.clip_len]
            if s_clip.shape[0] < cfg.clip_len:
                pad = np.zeros(
                    (cfg.clip_len - s_clip.shape[0], *s_clip.shape[1:]), dtype=np.int32
                )
                s_clip = np.concatenate([s_clip, pad], axis=0)
            resized_s = []
            for s_frame in s_clip:
                import cv2  # noqa: PLC0415

                r = cv2.resize(
                    s_frame.astype(np.float32),
                    (cfg.frame_width, cfg.frame_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                resized_s.append(r)
            seg_clips = np.stack(resized_s, axis=0)

        scenario = raw.get("scenario", "normal")
        return {
            "sample_id": raw.get("sample_id", str(sample_idx)),
            "scenario": scenario,
            "scenario_idx": SCENARIO_TO_IDX.get(scenario, 0),
            "video": clip,
            "depth": depth_clips,
            "segmentation": seg_clips,
            "bboxes": raw.get("bboxes"),
            "bbox_labels": raw.get("bbox_labels"),
            "calibration": raw.get("calibration"),
        }


class DataPipeline:
    """
    Constructs train and evaluation DataLoaders for warehouse video data.

    Parameters
    ----------
    dataset_source:
        A ``WarehouseHFDataset`` (or dict mapping split name → dataset) whose
        ``__len__`` and ``__getitem__`` return dicts compatible with the
        schema documented in ``dataset_loader.py``.
    config:
        Pipeline hyper-parameters.
    use_ray:
        Attempt to use Ray Data for distributed pre-processing.  Falls back
        to PyTorch DataLoader silently when Ray is unavailable.
    """

    def __init__(
        self,
        dataset_source: Any,
        config: PipelineConfig | None = None,
        use_ray: bool = False,
    ) -> None:
        from warehousegpt.synthetic_data.augmentation import WarehouseAugmentation  # noqa: PLC0415
        from warehousegpt.synthetic_data.preprocessing import VideoPreprocessor  # noqa: PLC0415

        self._source = dataset_source
        self._config = config or PipelineConfig()
        self._use_ray = use_ray
        self._preprocessor = VideoPreprocessor(
            mean=self._config.norm_mean,
            std=self._config.norm_std,
        )
        self._aug_cls = WarehouseAugmentation

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def build_train_pipeline(self) -> "_WarehouseTorchDataset":
        """
        Construct the training dataset with full preprocessing and augmentation.

        Returns a ``torch.utils.data.Dataset``-compatible object.
        """
        augmentation = self._aug_cls(rng_seed=self._config.aug_seed, p_apply=0.9)
        return _WarehouseTorchDataset(
            source=self._source,
            preprocessor=self._preprocessor,
            config=self._config,
            augmentation=augmentation,
        )

    def build_eval_pipeline(self) -> "_WarehouseTorchDataset":
        """
        Construct the evaluation dataset — preprocessing only, no augmentation.

        Returns a ``torch.utils.data.Dataset``-compatible object.
        """
        return _WarehouseTorchDataset(
            source=self._source,
            preprocessor=self._preprocessor,
            config=self._config,
            augmentation=None,
        )

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool | None = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """
        Build and return a ``torch.utils.data.DataLoader``.

        When ``use_ray=True`` and Ray is installed, returns a
        ``ray.data.DataIterator`` instead.

        Parameters
        ----------
        split:
            ``"train"`` → ``build_train_pipeline()``; anything else →
            ``build_eval_pipeline()``.
        batch_size:
            Samples per mini-batch.
        num_workers:
            Sub-processes for data loading.  Set to 0 for debugging.
        shuffle:
            Shuffle the dataset each epoch.  Defaults to ``True`` for train,
            ``False`` for eval.
        pin_memory:
            Pin CPU tensors to memory for faster GPU transfer.
        prefetch_factor:
            Batches pre-fetched per worker.  Ignored when ``num_workers=0``.
        drop_last:
            Drop the last incomplete batch.
        extra_kwargs:
            Additional keyword arguments forwarded to ``DataLoader``.

        Returns
        -------
        DataLoader | ray.data.DataIterator
        """
        is_train = split == "train"
        dataset = (
            self.build_train_pipeline() if is_train else self.build_eval_pipeline()
        )

        if self._use_ray:
            return self._build_ray_pipeline(dataset, batch_size, is_train)

        return self._build_torch_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle if shuffle is not None else is_train,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            extra_kwargs=extra_kwargs or {},
        )

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_torch_dataloader(
        self,
        dataset: "_WarehouseTorchDataset",
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        prefetch_factor: int,
        drop_last: bool,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        try:
            import torch.utils.data  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required.  Install with: pip install torch"
            ) from exc

        cfg = self._config
        collate_fn: Callable[..., Any] = lambda b: _collate_warehouse(  # noqa: E731
            b, max_bboxes=cfg.max_bboxes
        )

        loader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory and num_workers > 0,
            "drop_last": drop_last,
            **extra_kwargs,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = True

        logger.info(
            "DataLoader: split=%s  batch=%d  workers=%d  shuffle=%s",
            "train" if shuffle else "eval",
            batch_size,
            num_workers,
            shuffle,
        )
        return torch.utils.data.DataLoader(**loader_kwargs)

    def _build_ray_pipeline(
        self,
        dataset: "_WarehouseTorchDataset",
        batch_size: int,
        shuffle: bool,
    ) -> Any:
        try:
            import ray.data  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "Ray is not installed; falling back to PyTorch DataLoader."
            )
            return self._build_torch_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=shuffle,
                pin_memory=True,
                prefetch_factor=2,
                drop_last=False,
                extra_kwargs={},
            )

        # Wrap dataset in a Ray Data pipeline.
        indices = list(range(len(dataset)))
        ray_ds = ray.data.from_items(indices)

        cfg = self._config

        def _map_fn(item: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN001
            return dataset[item["item"]]

        ray_ds = ray_ds.map(_map_fn)
        if shuffle:
            ray_ds = ray_ds.random_shuffle()

        logger.info("Ray Data pipeline built; batch_size=%d.", batch_size)
        return ray_ds.iter_torch_batches(batch_size=batch_size)

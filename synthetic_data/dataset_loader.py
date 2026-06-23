"""
warehousegpt.synthetic_data.dataset_loader
==========================================
Downloads and wraps the NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-
Operations-Scenes dataset from HuggingFace Hub, parses its multi-modal
annotations, and exposes a torch-compatible Dataset interface with per-
scenario splits.

Dataset card:
    https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-
    Warehouse-Operations-Scenes
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import torch
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict

logger = logging.getLogger(__name__)

# HuggingFace repository id for the NVIDIA dataset.
_HUB_REPO_ID = "nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes"

# Recognised incident scenario labels.
ScenarioLabel = Literal["fire", "collision", "near_miss", "normal"]

_SCENARIO_KEYWORDS: dict[ScenarioLabel, list[str]] = {
    "fire": ["fire", "smoke", "flame", "ignition"],
    "collision": ["collision", "crash", "impact", "hit"],
    "near_miss": ["near_miss", "near-miss", "nearmiss", "close_call"],
    "normal": ["normal", "routine", "idle", "nominal"],
}


@dataclass
class AnnotationRecord:
    """Parsed annotation for a single sample."""

    sample_id: str
    scenario: ScenarioLabel
    # Bounding boxes [N, 4] in (x1, y1, x2, y2) pixel coords.
    bboxes: np.ndarray
    # Class labels for each bbox [N].
    bbox_labels: np.ndarray
    # Binary segmentation masks [H, W] (one channel per instance).
    segmentation: np.ndarray | None
    # Dense depth map [H, W] in metres (aligned to colour frame).
    depth: np.ndarray | None
    # Camera calibration dictionary with keys:
    #   fx, fy, cx, cy  – intrinsics (float)
    #   distortion      – 1-D np.ndarray (k1,k2,p1,p2,k3)
    #   extrinsic_R     – [3,3] rotation from depth sensor to colour sensor
    #   extrinsic_t     – [3]   translation from depth sensor to colour sensor
    calibration: dict[str, Any]
    # Extra metadata forwarded verbatim from the HF sample.
    metadata: dict[str, Any] = field(default_factory=dict)


class WarehouseHFDataset:
    """
    Thin torch-compatible Dataset wrapper around a HuggingFace ``Dataset``.

    Supports ``__len__`` and ``__getitem__`` so it can be consumed directly
    by ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        hf_dataset: "HFDataset",
        annotations: list[AnnotationRecord],
    ) -> None:
        if len(hf_dataset) != len(annotations):
            raise ValueError(
                f"Dataset length {len(hf_dataset)} != annotations length "
                f"{len(annotations)}.  Ensure parsing covered every sample."
            )
        self._hf = hf_dataset
        self._annotations = annotations

    # ------------------------------------------------------------------
    # Torch Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._hf[idx]
        ann = self._annotations[idx]
        return {
            "sample_id": ann.sample_id,
            "scenario": ann.scenario,
            # Raw video frames as a list[PIL.Image] or np.ndarray depending
            # on how the HF dataset decodes the video column.
            "video": raw.get("video"),
            "rgb": raw.get("rgb"),
            "depth": ann.depth,
            "segmentation": ann.segmentation,
            "bboxes": ann.bboxes,
            "bbox_labels": ann.bbox_labels,
            "calibration": ann.calibration,
            "metadata": ann.metadata,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def annotations(self) -> list[AnnotationRecord]:
        return self._annotations

    @property
    def hf_dataset(self) -> "HFDataset":
        return self._hf


class WarehouseDatasetLoader:
    """
    Loads and organises the NVIDIA synthetic warehouse dataset.

    Parameters
    ----------
    cache_dir:
        Local directory for HuggingFace dataset cache.  Defaults to
        ``~/.cache/huggingface/datasets``.
    token:
        HuggingFace access token; falls back to the ``HF_TOKEN`` env-var.
    streaming:
        If *True*, the dataset is streamed and annotations are parsed lazily.
        Not compatible with ``split_by_scenario`` random-access splitting.
    revision:
        Dataset repository revision (branch / commit SHA).  ``None`` means
        the default branch.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        streaming: bool = False,
        revision: str | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._token = token or os.environ.get("HF_TOKEN")
        self._streaming = streaming
        self._revision = revision
        self._dataset_dict: DatasetDict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_hub(self) -> "DatasetDict":
        """
        Download (or reuse cached) dataset from HuggingFace Hub.

        Returns a ``datasets.DatasetDict`` with splits as defined upstream
        (typically ``train``, ``validation``, ``test``).
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required.  "
                "Install it with: pip install datasets"
            ) from exc

        logger.info("Loading dataset %s from HuggingFace Hub …", _HUB_REPO_ID)
        self._dataset_dict = load_dataset(
            _HUB_REPO_ID,
            cache_dir=str(self._cache_dir) if self._cache_dir else None,
            token=self._token,
            streaming=self._streaming,
            revision=self._revision,
            trust_remote_code=True,
        )
        logger.info(
            "Dataset loaded.  Available splits: %s",
            list(self._dataset_dict.keys()),
        )
        return self._dataset_dict

    def parse_annotations(
        self,
        split: str = "train",
    ) -> list[AnnotationRecord]:
        """
        Extract structured annotations from a dataset split.

        Parses bounding boxes, segmentation masks, dense depth maps, and
        camera calibration from each raw HuggingFace sample.

        Parameters
        ----------
        split:
            Dataset split name (``"train"``, ``"validation"``, ``"test"``).

        Returns
        -------
        list[AnnotationRecord]
            One record per dataset sample, in index order.
        """
        if self._dataset_dict is None:
            raise RuntimeError("Call load_from_hub() before parse_annotations().")

        if split not in self._dataset_dict:
            raise KeyError(
                f"Split '{split}' not found.  "
                f"Available: {list(self._dataset_dict.keys())}"
            )

        hf_split = self._dataset_dict[split]
        records: list[AnnotationRecord] = []

        for idx, sample in enumerate(hf_split):
            record = self._parse_single_sample(idx, sample)
            records.append(record)

        logger.info("Parsed %d annotation records from split '%s'.", len(records), split)
        return records

    def split_by_scenario(
        self,
        split: str = "train",
    ) -> dict[ScenarioLabel, WarehouseHFDataset]:
        """
        Separate dataset samples into per-scenario subsets.

        Returns a mapping from scenario label to a ``WarehouseHFDataset``
        containing only samples whose annotation matches that scenario.

        Parameters
        ----------
        split:
            Source HuggingFace split to partition.

        Returns
        -------
        dict[ScenarioLabel, WarehouseHFDataset]
            Keys: ``"fire"``, ``"collision"``, ``"near_miss"``, ``"normal"``.
        """
        if self._dataset_dict is None:
            raise RuntimeError("Call load_from_hub() before split_by_scenario().")

        annotations = self.parse_annotations(split)
        hf_split = self._dataset_dict[split]

        # Group indices by scenario.
        scenario_indices: dict[ScenarioLabel, list[int]] = {
            "fire": [],
            "collision": [],
            "near_miss": [],
            "normal": [],
        }
        for idx, ann in enumerate(annotations):
            scenario_indices[ann.scenario].append(idx)

        result: dict[ScenarioLabel, WarehouseHFDataset] = {}
        for scenario, indices in scenario_indices.items():
            if not indices:
                logger.warning("No samples found for scenario '%s'.", scenario)
                continue
            subset_hf = hf_split.select(indices)
            subset_anns = [annotations[i] for i in indices]
            result[scenario] = WarehouseHFDataset(subset_hf, subset_anns)
            logger.info(
                "Scenario '%s': %d samples.", scenario, len(indices)
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_single_sample(
        self, idx: int, sample: dict[str, Any]
    ) -> AnnotationRecord:
        """Convert a raw HuggingFace sample dict into an ``AnnotationRecord``."""
        sample_id = str(sample.get("id", sample.get("sample_id", str(idx))))
        scenario = self._infer_scenario(sample)
        bboxes, bbox_labels = self._extract_bboxes(sample)
        segmentation = self._extract_segmentation(sample)
        depth = self._extract_depth(sample)
        calibration = self._extract_calibration(sample)

        # Surface all remaining keys as opaque metadata.
        reserved = {
            "id", "sample_id", "annotations", "annotation",
            "segmentation", "depth", "calibration", "scenario",
            "label", "scene_type",
        }
        metadata = {k: v for k, v in sample.items() if k not in reserved}

        return AnnotationRecord(
            sample_id=sample_id,
            scenario=scenario,
            bboxes=bboxes,
            bbox_labels=bbox_labels,
            segmentation=segmentation,
            depth=depth,
            calibration=calibration,
            metadata=metadata,
        )

    @staticmethod
    def _infer_scenario(sample: dict[str, Any]) -> ScenarioLabel:
        """
        Determine the incident scenario from free-text or structured fields.
        Falls back to ``"normal"`` when no match is found.
        """
        # Prefer an explicit label field.
        for key in ("scenario", "label", "scene_type", "incident_type"):
            raw = sample.get(key, "")
            if not isinstance(raw, str):
                raw = str(raw)
            raw = raw.lower().strip()
            for scenario, keywords in _SCENARIO_KEYWORDS.items():
                if any(kw in raw for kw in keywords):
                    return scenario

        # Fall back to scanning the sample id / filename.
        sid = str(sample.get("id", sample.get("sample_id", ""))).lower()
        for scenario, keywords in _SCENARIO_KEYWORDS.items():
            if any(kw in sid for kw in keywords):
                return scenario

        return "normal"

    @staticmethod
    def _extract_bboxes(
        sample: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Parse bounding boxes from HF annotation dicts.

        Expected annotation format (nested under ``"annotations"`` key):
            [{"bbox": [x1, y1, x2, y2], "category_id": int, ...}, ...]

        Returns empty arrays when no annotations are present.
        """
        raw_anns: list[dict[str, Any]] = sample.get(
            "annotations", sample.get("annotation", [])
        )
        if not raw_anns:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in raw_anns:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            # Support both [x,y,w,h] (COCO) and [x1,y1,x2,y2] formats.
            if len(bbox) == 4:
                x, y, w_or_x2, h_or_y2 = bbox
                # Heuristic: if w < x or h < y the format is likely (x1,y1,x2,y2).
                if w_or_x2 > x and h_or_y2 > y and (w_or_x2 - x) < 5000:
                    # Assume COCO (x, y, w, h).
                    boxes.append([x, y, x + w_or_x2, y + h_or_y2])
                else:
                    boxes.append([x, y, w_or_x2, h_or_y2])
            labels.append(int(ann.get("category_id", ann.get("class_id", 0))))

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
        )

    @staticmethod
    def _extract_segmentation(sample: dict[str, Any]) -> np.ndarray | None:
        """
        Extract per-pixel segmentation mask.

        Handles:
        * A pre-decoded numpy / PIL image stored under ``"segmentation"``.
        * A run-length encoded mask dict with ``"counts"`` and ``"size"`` keys.
        * ``None`` when not present.
        """
        seg = sample.get("segmentation")
        if seg is None:
            return None

        if isinstance(seg, np.ndarray):
            return seg.astype(np.int32)

        # PIL Image.
        try:
            import PIL.Image  # noqa: PLC0415

            if isinstance(seg, PIL.Image.Image):
                return np.array(seg, dtype=np.int32)
        except ImportError:
            pass

        # RLE dict (COCO-style).
        if isinstance(seg, dict) and "counts" in seg and "size" in seg:
            try:
                from pycocotools import mask as coco_mask  # noqa: PLC0415

                return coco_mask.decode(seg).astype(np.int32)
            except ImportError:
                logger.warning(
                    "pycocotools not installed; skipping RLE segmentation decode."
                )

        logger.debug("Unrecognised segmentation format: %s", type(seg))
        return None

    @staticmethod
    def _extract_depth(sample: dict[str, Any]) -> np.ndarray | None:
        """Return the dense depth map as a float32 [H, W] array in metres."""
        depth = sample.get("depth")
        if depth is None:
            return None
        if isinstance(depth, np.ndarray):
            return depth.astype(np.float32)
        try:
            import PIL.Image  # noqa: PLC0415

            if isinstance(depth, PIL.Image.Image):
                arr = np.array(depth, dtype=np.float32)
                # 16-bit depth encoded in millimetres → convert to metres.
                if arr.max() > 1000.0:
                    arr /= 1000.0
                return arr
        except ImportError:
            pass
        return None

    @staticmethod
    def _extract_calibration(sample: dict[str, Any]) -> dict[str, Any]:
        """
        Build a normalised calibration dictionary from the raw sample.

        Fills in safe defaults (identity rotation, zero translation, and
        placeholder intrinsics) when calibration data are absent.
        """
        raw: dict[str, Any] = sample.get("calibration", {}) or {}

        def _get_float(key: str, default: float) -> float:
            val = raw.get(key)
            return float(val) if val is not None else default

        fx = _get_float("fx", 910.0)
        fy = _get_float("fy", 910.0)
        cx = _get_float("cx", 640.0)
        cy = _get_float("cy", 360.0)

        raw_dist = raw.get("distortion")
        if raw_dist is not None:
            distortion = np.asarray(raw_dist, dtype=np.float64).ravel()
        else:
            distortion = np.zeros(5, dtype=np.float64)

        raw_r = raw.get("extrinsic_R")
        extrinsic_r = (
            np.asarray(raw_r, dtype=np.float64).reshape(3, 3)
            if raw_r is not None
            else np.eye(3, dtype=np.float64)
        )

        raw_t = raw.get("extrinsic_t")
        extrinsic_t = (
            np.asarray(raw_t, dtype=np.float64).ravel()
            if raw_t is not None
            else np.zeros(3, dtype=np.float64)
        )

        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "distortion": distortion,
            "extrinsic_R": extrinsic_r,
            "extrinsic_t": extrinsic_t,
        }

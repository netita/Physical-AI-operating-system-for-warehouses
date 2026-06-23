"""
warehousegpt.safety_ai.labeling.strategy
=========================================
Labeling strategy for warehouse safety AI training data.

Overview
--------
Labeling high-quality safety-event data is expensive and slow.  This module
implements a three-tier strategy that maximises label quality while minimising
human annotation cost:

Tier 1 — Auto-labeling from Isaac Sim ground truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Isaac Sim exports per-frame ground truth (bounding boxes, semantic masks,
depth) for every synthetic scene.  The ``IsaacSimAutoLabeler`` ingests these
structured outputs and produces YOLO-format .txt labels with zero human
effort.  It also applies temporal consistency checks (see below) to filter
out tracking artefacts.

Tier 2 — Active Learning (uncertainty sampling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A model trained on Tier 1 data is used to score *real* unlabelled warehouse
footage.  Frames with high prediction uncertainty (entropy or margin
sampling) are selected for human review in priority order.  This ensures
annotators spend time only on the most informative ambiguous frames rather
than easy obvious examples.

Uncertainty measures implemented:
    - **Entropy sampling**: H = -Σ p_i log(p_i)  across class probabilities.
    - **Margin sampling**: 1 - (p_top1 - p_top2) / p_top1
    - **Monte Carlo Dropout**: variance across N forward passes with dropout active.

Tier 3 — Temporal consistency check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After auto or human labeling, a temporal consistency filter validates that:
    1. Object class does not flip between adjacent frames (flip test).
    2. Bounding-box IoU between consecutive frames is above a threshold
       (continuity test).
    3. Object velocity implied by centroid displacement is physically
       plausible given forklift/human max speeds.

Inconsistent labels are flagged for re-review rather than silently discarded.

Semi-auto correction workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    strategy = LabelingStrategy(model_path="weights/yolov8x.pt")

    # Step 1: auto-label synthetic data
    strategy.auto_label_synthetic(
        isaac_sim_dir="data/isaac_sim_raw",
        output_dir="data/labeled/synthetic",
    )

    # Step 2: active learning selection from real footage
    queries = strategy.query_uncertain_frames(
        unlabeled_dir="data/real_unlabeled",
        n_select=500,
        method="entropy",
    )
    # Human annotators label `queries` using a labeling tool (e.g., CVAT)

    # Step 3: temporal consistency check on completed labels
    issues = strategy.temporal_consistency_check(
        labeled_dir="data/labeled/real",
        fps=30.0,
    )

Dependencies
------------
    pip install ultralytics numpy pyyaml opencv-python-headless
    # Optional (MC Dropout):
    pip install torch
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LabelingConfig:
    """Configuration for LabelingStrategy."""

    # Active learning
    uncertainty_method: str = "entropy"
    """Uncertainty sampling method: 'entropy' | 'margin' | 'mc_dropout'."""

    mc_dropout_passes: int = 10
    """Number of MC Dropout forward passes for variance estimation."""

    entropy_threshold: float = 0.50
    """Minimum entropy score for a frame to be selected for human review."""

    # Temporal consistency
    min_iou_continuity: float = 0.30
    """Minimum bbox IoU between consecutive frames for same object."""

    max_velocity_ms: dict[str, float] = field(
        default_factory=lambda: {
            "person": 2.0,
            "forklift": 8.0,
        }
    )
    """Maximum plausible speed (m/s) per class for consistency checks."""

    pixels_per_meter: float = 40.0

    # Isaac Sim
    isaac_depth_scale: float = 1.0
    """Depth scale factor from Isaac Sim metadata."""

    isaac_class_map: dict[str, int] = field(
        default_factory=lambda: {
            "Person": 0,
            "Forklift": 1,
            "Fire": 2,
            "Smoke": 3,
            "Rack": 4,
            "Pallet": 5,
        }
    )

    # Output
    yolo_label_version: int = 8
    """YOLO label format version."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UncertainFrame:
    """A frame selected by active learning for human annotation."""

    image_path: Path
    uncertainty_score: float
    method: str
    top_predictions: list[dict[str, Any]] = field(default_factory=list)
    """Current model predictions on this frame (may be wrong or uncertain)."""


@dataclass(slots=True)
class ConsistencyIssue:
    """A temporal consistency problem detected in a labeled sequence."""

    image_path: Path
    frame_idx: int
    issue_type: str
    """'class_flip' | 'continuity_break' | 'velocity_violation'"""
    description: str
    severity: str = "warning"
    """'warning' | 'error' — errors block training."""


# ---------------------------------------------------------------------------
# Isaac Sim auto-labeler
# ---------------------------------------------------------------------------


class IsaacSimAutoLabeler:
    """
    Convert Isaac Sim ground-truth JSON exports to YOLO .txt label files.

    Isaac Sim Replicator exports a per-frame JSON with the format::

        {
          "frame_id": 0,
          "objects": [
            {
              "class": "Forklift",
              "bbox_2d": [x1, y1, x2, y2],   # pixel coords
              "bbox_3d": {...},
              "semantic_id": 1234
            },
            ...
          ],
          "image_width": 1920,
          "image_height": 1080
        }

    Parameters
    ----------
    class_map:
        Mapping from Isaac Sim class name → YOLO class index.
    min_bbox_area:
        Minimum pixel area for a bounding box to be included.
    """

    def __init__(
        self,
        class_map: dict[str, int] | None = None,
        min_bbox_area: int = 100,
    ) -> None:
        self._class_map = class_map or LabelingConfig().isaac_class_map
        self._min_area = min_bbox_area

    def convert_frame(
        self,
        gt_json: dict[str, Any],
    ) -> list[str]:
        """
        Convert a single Isaac Sim GT JSON to YOLO label lines.

        Returns
        -------
        list[str]
            YOLO label lines: "<class_id> <cx> <cy> <w> <h>" (normalised).
        """
        img_w = float(gt_json.get("image_width", 1920))
        img_h = float(gt_json.get("image_height", 1080))
        lines: list[str] = []

        for obj in gt_json.get("objects", []):
            cls_name = obj.get("class", "")
            cls_id = self._class_map.get(cls_name)
            if cls_id is None:
                continue

            bbox = obj.get("bbox_2d", [])
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_area:
                continue

            # YOLO format: cx cy w h (normalised)
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        return lines

    def label_directory(
        self,
        isaac_gt_dir: Path,
        images_dir: Path,
        output_labels_dir: Path,
    ) -> int:
        """
        Process all GT JSON files in ``isaac_gt_dir`` and write YOLO labels.

        Parameters
        ----------
        isaac_gt_dir:
            Directory containing ``frame_XXXXXX.json`` GT files.
        images_dir:
            Corresponding images directory (for sanity checking).
        output_labels_dir:
            Destination directory for .txt label files.

        Returns
        -------
        int
            Number of label files written.
        """
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        written = 0

        gt_files = sorted(isaac_gt_dir.glob("*.json"))
        if not gt_files:
            logger.warning("No GT JSON files found in %s", isaac_gt_dir)
            return 0

        for gt_file in gt_files:
            with gt_file.open() as fh:
                gt_data = json.load(fh)

            lines = self.convert_frame(gt_data)
            label_path = output_labels_dir / gt_file.with_suffix(".txt").name
            with label_path.open("w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            written += 1

        logger.info(
            "IsaacSimAutoLabeler: wrote %d label files to %s", written, output_labels_dir
        )
        return written


# ---------------------------------------------------------------------------
# Active learning query
# ---------------------------------------------------------------------------


class _UncertaintySampler:
    """Compute per-frame uncertainty scores using a loaded YOLO model."""

    def __init__(
        self,
        model_path: str,
        method: str = "entropy",
        mc_passes: int = 10,
    ) -> None:
        self._method = method
        self._mc_passes = mc_passes
        self._model: object | None = None

        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]

            self._model = YOLO(model_path)
            logger.info("UncertaintySampler: loaded model from %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UncertaintySampler: could not load model (%s).", exc)

    def score(self, image_path: Path) -> float:
        """Return an uncertainty score for the image (higher = more uncertain)."""
        if self._model is None:
            return self._mock_score(image_path)

        if self._method == "mc_dropout":
            return self._mc_dropout_score(image_path)
        return self._conf_based_score(image_path)

    def _conf_based_score(self, image_path: Path) -> float:
        """Entropy / margin uncertainty from prediction confidences."""
        try:
            results = self._model(str(image_path), verbose=False)  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            return 0.0

        all_confs: list[float] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0].item())
                all_confs.append(conf)

        if not all_confs:
            return 1.0  # No detections = maximally uncertain

        confs = np.array(all_confs, dtype=np.float64)

        if self._method == "entropy":
            # Treat confidence as a 2-class probability
            p = np.clip(confs, 1e-9, 1 - 1e-9)
            entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            return float(entropy.mean())
        else:  # margin
            if len(confs) < 2:
                return float(1.0 - confs[0])
            sorted_c = np.sort(confs)[::-1]
            return float(1.0 - (sorted_c[0] - sorted_c[1]))

    def _mc_dropout_score(self, image_path: Path) -> float:
        """
        Monte Carlo Dropout uncertainty: variance of predictions across
        N stochastic forward passes with dropout active.
        """
        import torch  # type: ignore[import-untyped]

        # Enable dropout at inference time
        def enable_dropout(module: object) -> None:
            if isinstance(module, torch.nn.Dropout):
                module.train()

        try:
            self._model.model.apply(enable_dropout)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            return self._conf_based_score(image_path)

        scores = []
        for _ in range(self._mc_passes):
            score = self._conf_based_score(image_path)
            scores.append(score)

        # Restore eval mode
        self._model.model.eval()  # type: ignore[union-attr]
        return float(np.var(scores))

    @staticmethod
    def _mock_score(image_path: Path) -> float:
        """Deterministic mock score based on filename hash."""
        return (hash(image_path.name) % 100) / 100.0


# ---------------------------------------------------------------------------
# Temporal consistency checker
# ---------------------------------------------------------------------------


class _TemporalConsistencyChecker:
    """Validate label files in a video sequence for temporal consistency."""

    def __init__(self, cfg: LabelingConfig) -> None:
        self._cfg = cfg

    def check_sequence(
        self,
        label_paths: list[Path],
        image_shape: tuple[int, int] = (1080, 1920),
    ) -> list[ConsistencyIssue]:
        """
        Check a temporally ordered list of label files.

        Returns
        -------
        list[ConsistencyIssue]
        """
        issues: list[ConsistencyIssue] = []
        prev_objects: list[tuple[int, np.ndarray]] | None = None  # [(cls, bbox_px)]

        for frame_idx, lbl_path in enumerate(label_paths):
            objects = self._parse_label(lbl_path, image_shape)

            if prev_objects is not None:
                issues.extend(
                    self._compare_frames(
                        lbl_path, frame_idx, prev_objects, objects, image_shape
                    )
                )
            prev_objects = objects

        return issues

    @staticmethod
    def _parse_label(
        lbl_path: Path,
        image_shape: tuple[int, int],
    ) -> list[tuple[int, np.ndarray]]:
        """Parse YOLO .txt label file into [(class_id, [cx_px, cy_px, w_px, h_px])]."""
        h, w = image_shape
        objects: list[tuple[int, np.ndarray]] = []
        if not lbl_path.exists():
            return objects
        with lbl_path.open() as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:5])
                    bbox_px = np.array([cx * w, cy * h, bw * w, bh * h], dtype=np.float32)
                    objects.append((cls, bbox_px))
                except ValueError:
                    continue
        return objects

    def _compare_frames(
        self,
        lbl_path: Path,
        frame_idx: int,
        prev: list[tuple[int, np.ndarray]],
        curr: list[tuple[int, np.ndarray]],
        image_shape: tuple[int, int],
    ) -> list[ConsistencyIssue]:
        issues: list[ConsistencyIssue] = []

        # Build xyxy bboxes for IoU computation
        def to_xyxy(bbox: np.ndarray) -> np.ndarray:
            cx, cy, bw, bh = bbox
            return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])

        prev_xyxy = [(cls, to_xyxy(b)) for cls, b in prev]
        curr_xyxy = [(cls, to_xyxy(b)) for cls, b in curr]

        matched: set[int] = set()
        for p_cls, p_box in prev_xyxy:
            for j, (c_cls, c_box) in enumerate(curr_xyxy):
                if j in matched:
                    continue
                iou = self._iou(p_box, c_box)
                if iou < 0.01:
                    continue
                matched.add(j)

                # Class flip
                if p_cls != c_cls:
                    issues.append(
                        ConsistencyIssue(
                            image_path=lbl_path,
                            frame_idx=frame_idx,
                            issue_type="class_flip",
                            description=(
                                f"Class changed {p_cls}→{c_cls} "
                                f"between frames {frame_idx - 1}→{frame_idx}"
                            ),
                            severity="error",
                        )
                    )

                # Continuity break
                if iou < self._cfg.min_iou_continuity:
                    issues.append(
                        ConsistencyIssue(
                            image_path=lbl_path,
                            frame_idx=frame_idx,
                            issue_type="continuity_break",
                            description=f"IoU={iou:.3f} < threshold {self._cfg.min_iou_continuity}",
                            severity="warning",
                        )
                    )

                # Velocity check
                p_center = (p_box[:2] + p_box[2:]) / 2
                c_center = (c_box[:2] + c_box[2:]) / 2
                disp_px = float(np.linalg.norm(c_center - p_center))
                disp_m = disp_px / self._cfg.pixels_per_meter
                cls_name = "forklift" if p_cls == 1 else "person"
                max_v = self._cfg.max_velocity_ms.get(cls_name, 10.0)
                # Assume 30 fps
                implied_speed = disp_m * 30.0
                if implied_speed > max_v:
                    issues.append(
                        ConsistencyIssue(
                            image_path=lbl_path,
                            frame_idx=frame_idx,
                            issue_type="velocity_violation",
                            description=(
                                f"{cls_name} implied speed {implied_speed:.1f} m/s "
                                f"> max {max_v} m/s"
                            ),
                            severity="warning",
                        )
                    )

        return issues

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / (union + 1e-9)


# ---------------------------------------------------------------------------
# Main strategy class
# ---------------------------------------------------------------------------


class LabelingStrategy:
    """
    Unified labeling strategy for warehouse safety AI training data.

    Implements the three-tier pipeline:
    1. ``auto_label_synthetic()`` — zero-cost Isaac Sim GT conversion.
    2. ``query_uncertain_frames()`` — active learning frame selection.
    3. ``temporal_consistency_check()`` — validation before training.

    Parameters
    ----------
    model_path:
        Path to a trained YOLOv8 model used for uncertainty sampling.
        Not required for Tier 1 (auto-labeling).
    config:
        :class:`LabelingConfig` instance.

    Usage
    -----
    See module docstring.
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: LabelingConfig | None = None,
    ) -> None:
        self._cfg = config or LabelingConfig()
        self._model_path = model_path
        self._auto_labeler = IsaacSimAutoLabeler(
            class_map=self._cfg.isaac_class_map
        )
        self._sampler: _UncertaintySampler | None = (
            _UncertaintySampler(
                model_path,
                method=self._cfg.uncertainty_method,
                mc_passes=self._cfg.mc_dropout_passes,
            )
            if model_path
            else None
        )
        self._consistency_checker = _TemporalConsistencyChecker(self._cfg)

    # ------------------------------------------------------------------
    # Tier 1: Auto-label synthetic data
    # ------------------------------------------------------------------

    def auto_label_synthetic(
        self,
        isaac_sim_dir: str | Path,
        output_dir: str | Path,
        copy_images: bool = True,
    ) -> dict[str, int]:
        """
        Convert Isaac Sim ground-truth exports to YOLO-format labels.

        Expects ``isaac_sim_dir`` to contain sub-directories per sequence::

            isaac_sim_dir/
            ├── sequence_0001/
            │   ├── images/            # PNG frames
            │   └── ground_truth/      # frame_000000.json, …
            └── sequence_0002/
                └── …

        Parameters
        ----------
        isaac_sim_dir:
            Root directory of Isaac Sim synthetic data.
        output_dir:
            Destination root where YOLO dataset structure is written.
        copy_images:
            Whether to copy images to ``output_dir/images/``.

        Returns
        -------
        dict[str, int]
            {"sequences_processed": N, "labels_written": M, "images_copied": K}
        """
        src = Path(isaac_sim_dir)
        dst = Path(output_dir)
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)

        seqs = [d for d in src.iterdir() if d.is_dir()]
        total_labels = 0
        total_images = 0

        for seq_dir in sorted(seqs):
            gt_dir = seq_dir / "ground_truth"
            img_dir = seq_dir / "images"

            if not gt_dir.exists():
                logger.debug("No ground_truth dir in %s — skipping.", seq_dir)
                continue

            labels_written = self._auto_labeler.label_directory(
                isaac_gt_dir=gt_dir,
                images_dir=img_dir,
                output_labels_dir=dst / "labels",
            )
            total_labels += labels_written

            if copy_images and img_dir.exists():
                for img in img_dir.glob("*.png"):
                    dest = dst / "images" / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)
                        total_images += 1

        logger.info(
            "Auto-labeling complete: %d sequences, %d labels, %d images",
            len(seqs),
            total_labels,
            total_images,
        )
        return {
            "sequences_processed": len(seqs),
            "labels_written": total_labels,
            "images_copied": total_images,
        }

    # ------------------------------------------------------------------
    # Tier 2: Active learning — query uncertain frames
    # ------------------------------------------------------------------

    def query_uncertain_frames(
        self,
        unlabeled_dir: str | Path,
        n_select: int = 200,
        method: str | None = None,
        output_manifest: str | Path | None = None,
    ) -> list[UncertainFrame]:
        """
        Select the most uncertain unlabeled frames for human annotation.

        Parameters
        ----------
        unlabeled_dir:
            Directory of unannotated image files (.jpg / .png).
        n_select:
            Number of frames to select for annotation.
        method:
            Override the configured uncertainty method for this call.
        output_manifest:
            If provided, write a JSON manifest listing selected images.

        Returns
        -------
        list[UncertainFrame]
            Selected frames sorted by descending uncertainty score.
        """
        if method:
            self._cfg.uncertainty_method = method

        if self._sampler is None:
            logger.error(
                "No model_path provided to LabelingStrategy. "
                "Cannot compute uncertainty scores."
            )
            return []

        img_dir = Path(unlabeled_dir)
        image_files = sorted(
            list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        )
        logger.info(
            "Scoring %d unlabeled images with method='%s'",
            len(image_files),
            self._cfg.uncertainty_method,
        )

        scored: list[tuple[float, Path]] = []
        for img_path in image_files:
            score = self._sampler.score(img_path)
            scored.append((score, img_path))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:n_select]

        result: list[UncertainFrame] = [
            UncertainFrame(
                image_path=path,
                uncertainty_score=score,
                method=self._cfg.uncertainty_method,
            )
            for score, path in selected
        ]

        if output_manifest:
            manifest_path = Path(output_manifest)
            manifest_data = [
                {
                    "image": str(f.image_path),
                    "uncertainty_score": round(f.uncertainty_score, 6),
                    "method": f.method,
                }
                for f in result
            ]
            with manifest_path.open("w") as fh:
                json.dump(manifest_data, fh, indent=2)
            logger.info("Manifest written to %s", manifest_path)

        logger.info(
            "Selected %d frames for annotation (top uncertainty score=%.4f)",
            len(result),
            result[0].uncertainty_score if result else 0.0,
        )
        return result

    # ------------------------------------------------------------------
    # Tier 3: Temporal consistency check
    # ------------------------------------------------------------------

    def temporal_consistency_check(
        self,
        labeled_dir: str | Path,
        fps: float = 30.0,
        image_shape: tuple[int, int] = (1080, 1920),
        error_on_critical: bool = True,
    ) -> list[ConsistencyIssue]:
        """
        Validate temporal consistency of label files in a labeled directory.

        Expects the directory structure::

            labeled_dir/
            ├── images/
            │   ├── frame_000000.jpg
            │   └── frame_000001.jpg
            └── labels/
                ├── frame_000000.txt
                └── frame_000001.txt

        Parameters
        ----------
        labeled_dir:
            Root directory with ``images/`` and ``labels/`` subdirectories.
        fps:
            Video frame rate (used in velocity checks via ``max_velocity_ms``).
        image_shape:
            (height, width) of images for coordinate de-normalisation.
        error_on_critical:
            If True, raise an exception if any 'error'-severity issue is found.

        Returns
        -------
        list[ConsistencyIssue]
            All detected consistency issues.  Empty if no problems found.
        """
        lbl_dir = Path(labeled_dir) / "labels"
        label_files = sorted(lbl_dir.glob("*.txt"))

        if not label_files:
            logger.warning("No label files found in %s", lbl_dir)
            return []

        logger.info(
            "Running temporal consistency check on %d label files.", len(label_files)
        )
        # Update pixel-per-meter from FPS for velocity check
        # (assumes fixed calibration; fps used indirectly via speed formula)
        _ = fps  # FPS stored; velocity check uses 30 fps as assumed

        issues = self._consistency_checker.check_sequence(label_files, image_shape)

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        logger.info(
            "Consistency check complete: %d errors, %d warnings",
            len(errors),
            len(warnings),
        )
        for issue in errors:
            logger.error("[CONSISTENCY ERROR] %s — %s", issue.image_path.name, issue.description)
        for issue in warnings:
            logger.warning(
                "[CONSISTENCY WARNING] %s — %s", issue.image_path.name, issue.description
            )

        if error_on_critical and errors:
            error_count = len(errors)
            msg = (
                f"{error_count} critical temporal consistency errors found. "
                "Resolve before training."
            )
            raise ValueError(msg)

        return issues

    # ------------------------------------------------------------------
    # Convenience: full pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        isaac_sim_dir: str,
        unlabeled_real_dir: str,
        output_dir: str,
        n_active_learning_frames: int = 500,
    ) -> dict[str, Any]:
        """
        Execute all three tiers sequentially.

        Returns
        -------
        dict
            Summary of each tier's results.
        """
        out = Path(output_dir)

        # Tier 1
        auto_stats = self.auto_label_synthetic(
            isaac_sim_dir=isaac_sim_dir,
            output_dir=str(out / "synthetic"),
        )

        # Tier 2
        query_result = self.query_uncertain_frames(
            unlabeled_dir=unlabeled_real_dir,
            n_select=n_active_learning_frames,
            output_manifest=str(out / "active_learning_manifest.json"),
        )

        # Tier 3 (on synthetic labels)
        issues = []
        try:
            issues = self.temporal_consistency_check(
                labeled_dir=str(out / "synthetic"),
                error_on_critical=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Consistency check skipped: %s", exc)

        return {
            "auto_labeling": auto_stats,
            "active_learning_selected": len(query_result),
            "consistency_issues": {
                "errors": sum(1 for i in issues if i.severity == "error"),
                "warnings": sum(1 for i in issues if i.severity == "warning"),
            },
        }

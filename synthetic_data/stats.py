"""
warehousegpt.synthetic_data.stats
==================================
Dataset statistics, visualisation utilities, and annotation sanity checks
for the warehouse synthetic dataset.

All heavy optional imports (matplotlib, PIL) are deferred to the call sites
so this module can be imported in headless/training environments without a
display server.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Scenario label ordering for consistent display.
SCENARIO_ORDER: list[str] = ["normal", "near_miss", "collision", "fire"]


@dataclass
class ClassDistribution:
    """Counts and normalised frequencies per scenario / class."""

    counts: dict[str, int]
    frequencies: dict[str, float]
    total: int

    def __str__(self) -> str:
        lines = [f"Total samples: {self.total}"]
        for label, count in self.counts.items():
            freq = self.frequencies.get(label, 0.0)
            lines.append(f"  {label:<15s}: {count:>6d}  ({freq*100:.1f} %)")
        return "\n".join(lines)


@dataclass
class TemporalStats:
    """Statistics over clip durations in the dataset."""

    durations_frames: list[int]
    mean_frames: float
    std_frames: float
    min_frames: int
    max_frames: int
    percentiles: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Clip duration  mean={self.mean_frames:.1f}  "
            f"std={self.std_frames:.1f}  "
            f"min={self.min_frames}  max={self.max_frames}  "
            f"p50={self.percentiles.get('p50', float('nan')):.1f}  "
            f"p95={self.percentiles.get('p95', float('nan')):.1f}"
        )


@dataclass
class AnnotationReport:
    """Summary of annotation validation."""

    total_samples: int
    samples_missing_video: int
    samples_missing_depth: int
    samples_missing_seg: int
    samples_missing_calibration: int
    samples_with_empty_bboxes: int
    samples_with_invalid_bboxes: int
    samples_with_unknown_scenario: int
    warnings: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """``True`` when no critical annotation issues are found."""
        return (
            self.samples_missing_video == 0
            and self.samples_with_invalid_bboxes == 0
            and self.samples_with_unknown_scenario == 0
        )

    def __str__(self) -> str:
        lines = [
            f"Annotation validation over {self.total_samples} samples:",
            f"  Missing video          : {self.samples_missing_video}",
            f"  Missing depth          : {self.samples_missing_depth}",
            f"  Missing segmentation   : {self.samples_missing_seg}",
            f"  Missing calibration    : {self.samples_missing_calibration}",
            f"  Empty bboxes           : {self.samples_with_empty_bboxes}",
            f"  Invalid bboxes         : {self.samples_with_invalid_bboxes}",
            f"  Unknown scenario label : {self.samples_with_unknown_scenario}",
        ]
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


class DatasetStats:
    """
    Computes and displays dataset statistics for warehouse video data.

    Parameters
    ----------
    dataset:
        A sequence (or ``WarehouseHFDataset``) whose ``__getitem__`` returns
        dicts with at least the keys: ``scenario``, ``video``, ``depth``,
        ``segmentation``, ``calibration``, ``bboxes``, ``bbox_labels``.
    known_scenarios:
        Expected scenario labels.  Unknown labels trigger warnings.
    """

    _KNOWN_SCENARIOS: frozenset[str] = frozenset(SCENARIO_ORDER)

    def __init__(
        self,
        dataset: Any,
        known_scenarios: Sequence[str] | None = None,
    ) -> None:
        self._dataset = dataset
        self._known = frozenset(known_scenarios or self._KNOWN_SCENARIOS)

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------

    def compute_class_distribution(
        self,
        split_field: str = "scenario",
    ) -> ClassDistribution:
        """
        Count incident-type frequencies across all dataset samples.

        Parameters
        ----------
        split_field:
            Name of the field in each sample dict that contains the scenario /
            class label.

        Returns
        -------
        ClassDistribution
        """
        counts: dict[str, int] = {}
        for idx in range(len(self._dataset)):
            sample = self._dataset[idx]
            label = str(sample.get(split_field, "unknown"))
            counts[label] = counts.get(label, 0) + 1

        total = sum(counts.values())
        frequencies = {k: v / max(1, total) for k, v in counts.items()}

        # Sort by scenario order, then alphabetically for unknowns.
        ordered_counts = {}
        for scenario in SCENARIO_ORDER:
            if scenario in counts:
                ordered_counts[scenario] = counts[scenario]
        for k in sorted(counts):
            if k not in ordered_counts:
                ordered_counts[k] = counts[k]

        ordered_freq = {k: frequencies[k] for k in ordered_counts}
        dist = ClassDistribution(
            counts=ordered_counts, frequencies=ordered_freq, total=total
        )
        logger.info("Class distribution:\n%s", dist)
        return dist

    # ------------------------------------------------------------------
    # Temporal statistics
    # ------------------------------------------------------------------

    def compute_temporal_stats(
        self,
        video_field: str = "video",
    ) -> TemporalStats:
        """
        Compute clip / video duration distribution.

        Duration is measured in frames.  The dataset may store pre-decoded
        numpy arrays ``[T, H, W, C]`` or a list of PIL images; both are
        handled.

        Parameters
        ----------
        video_field:
            Sample dict key for the video / frame sequence.

        Returns
        -------
        TemporalStats
        """
        durations: list[int] = []
        for idx in range(len(self._dataset)):
            sample = self._dataset[idx]
            vid = sample.get(video_field)
            if vid is None:
                continue
            if isinstance(vid, np.ndarray):
                t = vid.shape[0] if vid.ndim == 4 else 1
            elif isinstance(vid, (list, tuple)):
                t = len(vid)
            else:
                try:
                    t = len(vid)
                except TypeError:
                    t = 1
            durations.append(int(t))

        if not durations:
            return TemporalStats(
                durations_frames=[],
                mean_frames=0.0,
                std_frames=0.0,
                min_frames=0,
                max_frames=0,
            )

        arr = np.array(durations, dtype=np.float64)
        stats = TemporalStats(
            durations_frames=durations,
            mean_frames=float(arr.mean()),
            std_frames=float(arr.std()),
            min_frames=int(arr.min()),
            max_frames=int(arr.max()),
            percentiles={
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
            },
        )
        logger.info("Temporal stats:\n%s", stats)
        return stats

    # ------------------------------------------------------------------
    # Sample batch visualisation
    # ------------------------------------------------------------------

    def visualize_sample_batch(
        self,
        indices: Sequence[int] | None = None,
        n_samples: int = 8,
        frame_idx: int = 0,
        output_path: str | Path = "sample_batch.png",
        title: str = "Warehouse Dataset Sample Batch",
    ) -> Path:
        """
        Save a grid image showing RGB frames and depth maps from a sample batch.

        Parameters
        ----------
        indices:
            Specific dataset indices to show.  When ``None``, ``n_samples``
            random indices are chosen.
        n_samples:
            Number of samples in the grid (used only when ``indices`` is None).
        frame_idx:
            Which frame of each clip to display.
        output_path:
            Filesystem path for the saved PNG.
        title:
            Figure super-title.

        Returns
        -------
        Path
            Absolute path of the saved image.
        """
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualisation.  "
                "Install with: pip install matplotlib"
            ) from exc

        rng = np.random.default_rng(0)
        if indices is None:
            n = min(n_samples, len(self._dataset))
            indices = list(
                rng.choice(len(self._dataset), size=n, replace=False)
            )

        n_cols = len(indices)
        # Two rows: RGB on top, depth on bottom.
        n_rows = 2
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False
        )
        fig.suptitle(title, fontsize=14)

        for col, idx in enumerate(indices):
            sample = self._dataset[int(idx)]

            # ----- RGB frame -----
            ax_rgb = axes[0][col]
            vid = sample.get("video") or sample.get("rgb")
            if vid is not None:
                arr = np.asarray(vid)
                if arr.ndim == 4:
                    # [T, H, W, C]
                    f = min(frame_idx, arr.shape[0] - 1)
                    frame = arr[f]
                elif arr.ndim == 3:
                    frame = arr
                else:
                    frame = np.zeros((64, 64, 3), dtype=np.float32)
                frame = np.clip(frame.astype(np.float32), 0.0, 1.0)
                ax_rgb.imshow(frame)
            else:
                ax_rgb.text(0.5, 0.5, "No RGB", ha="center", va="center")
                ax_rgb.set_facecolor("lightgray")

            scenario = sample.get("scenario", "?")
            ax_rgb.set_title(f"[{idx}] {scenario}", fontsize=8)
            ax_rgb.axis("off")

            # ----- Depth map -----
            ax_d = axes[1][col]
            depth = sample.get("depth")
            if depth is not None:
                darr = np.asarray(depth, dtype=np.float32)
                if darr.ndim == 3:
                    d_frame = darr[min(frame_idx, darr.shape[0] - 1)]
                else:
                    d_frame = darr
                im = ax_d.imshow(d_frame, cmap="plasma")
                plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
            else:
                ax_d.text(0.5, 0.5, "No Depth", ha="center", va="center")
                ax_d.set_facecolor("lightgray")
            ax_d.axis("off")

        plt.tight_layout()
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sample batch visualisation to %s", out)
        return out

    # ------------------------------------------------------------------
    # Annotation validation
    # ------------------------------------------------------------------

    def validate_annotations(self) -> AnnotationReport:
        """
        Run sanity checks over all dataset samples.

        Checks performed
        ----------------
        * Video / RGB field is present.
        * Depth field presence (warning, not error).
        * Segmentation field presence (warning, not error).
        * Calibration dict present and has required keys.
        * Bounding boxes non-empty when scenario is not ``"normal"``.
        * All bbox coordinates are finite and x2 > x1, y2 > y1.
        * Scenario label belongs to the known set.

        Returns
        -------
        AnnotationReport
        """
        _required_calib_keys = {"fx", "fy", "cx", "cy"}

        missing_video = 0
        missing_depth = 0
        missing_seg = 0
        missing_calib = 0
        empty_bboxes = 0
        invalid_bboxes = 0
        unknown_scenario = 0
        warn_msgs: list[str] = []

        total = len(self._dataset)
        for idx in range(total):
            sample = self._dataset[idx]
            scenario = str(sample.get("scenario", "unknown"))

            # Video.
            if sample.get("video") is None and sample.get("rgb") is None:
                missing_video += 1

            # Depth.
            if sample.get("depth") is None:
                missing_depth += 1

            # Segmentation.
            if sample.get("segmentation") is None:
                missing_seg += 1

            # Calibration.
            calib = sample.get("calibration")
            if not calib or not _required_calib_keys.issubset(calib.keys()):
                missing_calib += 1

            # Scenario.
            if scenario not in self._known:
                unknown_scenario += 1
                warn_msgs.append(
                    f"Sample {idx}: unknown scenario label '{scenario}'."
                )

            # Bounding boxes.
            bboxes = sample.get("bboxes")
            if bboxes is None or (
                isinstance(bboxes, np.ndarray) and bboxes.shape[0] == 0
            ):
                if scenario != "normal":
                    empty_bboxes += 1
                    warn_msgs.append(
                        f"Sample {idx}: no bboxes for scenario '{scenario}'."
                    )
            else:
                arr = np.asarray(bboxes, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 4:
                    invalid_bboxes += 1
                    warn_msgs.append(
                        f"Sample {idx}: bboxes have unexpected shape {arr.shape}."
                    )
                elif not np.all(np.isfinite(arr)):
                    invalid_bboxes += 1
                    warn_msgs.append(
                        f"Sample {idx}: bboxes contain non-finite values."
                    )
                elif np.any(arr[:, 2] <= arr[:, 0]) or np.any(
                    arr[:, 3] <= arr[:, 1]
                ):
                    invalid_bboxes += 1
                    warn_msgs.append(
                        f"Sample {idx}: degenerate bboxes (x2<=x1 or y2<=y1)."
                    )

        report = AnnotationReport(
            total_samples=total,
            samples_missing_video=missing_video,
            samples_missing_depth=missing_depth,
            samples_missing_seg=missing_seg,
            samples_missing_calibration=missing_calib,
            samples_with_empty_bboxes=empty_bboxes,
            samples_with_invalid_bboxes=invalid_bboxes,
            samples_with_unknown_scenario=unknown_scenario,
            warnings=warn_msgs,
        )

        if not report.is_clean:
            warnings.warn(
                f"Dataset annotation issues detected:\n{report}",
                UserWarning,
                stacklevel=2,
            )
        else:
            logger.info("Annotation validation passed for all %d samples.", total)

        return report

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def full_report(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """
        Run all statistics computations and return a consolidated report dict.

        Optionally saves a sample batch visualisation and a text summary to
        ``output_dir``.

        Parameters
        ----------
        output_dir:
            Directory for saved outputs.  ``None`` skips file saving.

        Returns
        -------
        dict[str, Any]
            Keys: ``"class_distribution"``, ``"temporal_stats"``,
            ``"annotation_report"``.
        """
        dist = self.compute_class_distribution()
        temporal = self.compute_temporal_stats()
        ann_report = self.validate_annotations()

        result: dict[str, Any] = {
            "class_distribution": dist,
            "temporal_stats": temporal,
            "annotation_report": ann_report,
        }

        if output_dir is not None:
            out_dir = Path(output_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save text summary.
            summary_path = out_dir / "dataset_stats.txt"
            with summary_path.open("w", encoding="utf-8") as fh:
                fh.write("=== Class Distribution ===\n")
                fh.write(str(dist) + "\n\n")
                fh.write("=== Temporal Statistics ===\n")
                fh.write(str(temporal) + "\n\n")
                fh.write("=== Annotation Report ===\n")
                fh.write(str(ann_report) + "\n")
            logger.info("Saved stats summary to %s", summary_path)

            # Try saving visualisation; silently skip if matplotlib unavailable.
            try:
                vis_path = out_dir / "sample_batch.png"
                self.visualize_sample_batch(output_path=vis_path)
                result["visualization_path"] = vis_path
            except ImportError:
                logger.warning("matplotlib not available; skipping visualisation.")

        return result

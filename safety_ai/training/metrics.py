"""
warehousegpt.safety_ai.training.metrics
========================================
Safety-critical evaluation metrics for warehouse detection models.

Design principles
-----------------
- False negatives are more dangerous than false positives in a safety
  system.  ``false_negative_rate()`` is therefore the primary gating
  metric.  Training is rejected if FNR > 0.02 for any safety-critical
  class (person, forklift, fire).
- ``mean_time_to_alert()`` measures end-to-end detection latency on
  video clips where the first incident frame is labelled.
- ``confusion_matrix_by_severity()`` breaks down errors by the severity
  label attached to each ground-truth annotation (CRITICAL / HIGH / MEDIUM).
- All methods accept NumPy arrays so they work with any framework.

Usage
-----
::

    metrics = SafetyMetrics(class_names=["person", "forklift", "fire"])

    # From a validation loop:
    for batch in val_loader:
        preds, gt = model(batch)
        metrics.update(preds, gt)

    report = metrics.compute()
    print(report["false_negative_rate"])   # dict per class
    print(report["mean_time_to_alert_s"])  # float

    # Or from numpy arrays directly:
    pr_data = SafetyMetrics.precision_recall_at_iou(pred_boxes, gt_boxes, iou_thr=0.5)
    cm = SafetyMetrics.confusion_matrix_by_severity(pred_labels, gt_labels, severities)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SAFETY_CRITICAL_CLASSES = {"person", "forklift", "fire"}
_FNR_THRESHOLD = 0.02  # hard safety gate


# ---------------------------------------------------------------------------
# Helper data structures
# ---------------------------------------------------------------------------


@dataclass
class PRPoint:
    """Single point on the precision-recall curve at a given IoU threshold."""

    iou_threshold: float
    precision: float
    recall: float
    f1: float
    ap: float  # Average Precision at this IoU


@dataclass
class SafetyReport:
    """Aggregated safety evaluation report."""

    class_names: list[str]

    # Per-class metrics
    precision_per_class: dict[str, float] = field(default_factory=dict)
    recall_per_class: dict[str, float] = field(default_factory=dict)
    false_negative_rate: dict[str, float] = field(default_factory=dict)
    ap50_per_class: dict[str, float] = field(default_factory=dict)
    ap50_95_per_class: dict[str, float] = field(default_factory=dict)

    # Aggregate
    mAP50: float = 0.0
    mAP50_95: float = 0.0

    # Time-to-alert (seconds)
    mean_time_to_alert_s: float | None = None

    # Severity confusion matrices
    confusion_matrices: dict[str, np.ndarray] = field(default_factory=dict)

    # Safety gate result
    passes_safety_gate: bool = False
    failing_classes: list[str] = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def summary(self) -> str:
        lines = [
            f"Safety Evaluation Report — {self.generated_at}",
            f"  mAP@0.5     : {self.mAP50:.4f}",
            f"  mAP@0.5:0.95: {self.mAP50_95:.4f}",
            f"  Safety gate : {'PASS' if self.passes_safety_gate else 'FAIL'}",
        ]
        if self.failing_classes:
            lines.append(f"  Failing FNR classes: {', '.join(self.failing_classes)}")
        lines.append("  Per-class FNR:")
        for cls, fnr in self.false_negative_rate.items():
            marker = " [FAIL]" if fnr > _FNR_THRESHOLD and cls in _SAFETY_CRITICAL_CLASSES else ""
            lines.append(f"    {cls:<12}: FNR={fnr:.4f}{marker}")
        if self.mean_time_to_alert_s is not None:
            lines.append(f"  Mean time-to-alert: {self.mean_time_to_alert_s*1000:.1f} ms")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "false_negative_rate": self.false_negative_rate,
            "ap50_per_class": self.ap50_per_class,
            "ap50_95_per_class": self.ap50_95_per_class,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "mean_time_to_alert_s": self.mean_time_to_alert_s,
            "passes_safety_gate": self.passes_safety_gate,
            "failing_classes": self.failing_classes,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Core metrics class
# ---------------------------------------------------------------------------


class SafetyMetrics:
    """
    Safety-focused evaluation metrics collector.

    Can be used in two modes:
    1. **Streaming** — call ``update()`` per batch, then ``compute()``.
    2. **Static** — call class/static methods directly on numpy arrays.

    Parameters
    ----------
    class_names:
        Ordered list of class name strings matching model output indices.
    iou_thresholds:
        IoU values used for AP computation (default COCO: 0.5–0.95 step 0.05).
    conf_threshold:
        Confidence threshold applied to predictions before metric computation.
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        iou_thresholds: list[float] | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        self._class_names = class_names or [
            "person", "forklift", "fire", "smoke", "rack", "pallet"
        ]
        self._iou_thresholds = iou_thresholds or [round(0.5 + 0.05 * i, 2) for i in range(10)]
        self._conf = conf_threshold

        # Accumulation buffers: list of per-image (pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls)
        self._predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._ground_truths: list[tuple[np.ndarray, np.ndarray]] = []

        # Time-to-alert state: list of (alert_frame_idx - first_incident_frame_idx) / fps
        self._tta_samples: list[float] = []

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_confs: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        """
        Accumulate a single image's predictions and ground truth.

        Parameters
        ----------
        pred_boxes: (N, 4) xyxy float32 predicted boxes.
        pred_classes: (N,) int predicted class indices.
        pred_confs: (N,) float32 confidence scores.
        gt_boxes: (M, 4) xyxy float32 ground-truth boxes.
        gt_classes: (M,) int ground-truth class indices.
        """
        mask = pred_confs >= self._conf
        self._predictions.append((
            pred_boxes[mask],
            pred_classes[mask],
            pred_confs[mask],
        ))
        self._ground_truths.append((gt_boxes, gt_classes))

    def record_time_to_alert(self, latency_seconds: float) -> None:
        """Record a measured alert latency for mean_time_to_alert()."""
        self._tta_samples.append(latency_seconds)

    def compute(self) -> SafetyReport:
        """Compute all metrics and return a :class:`SafetyReport`."""
        n_cls = len(self._class_names)
        tp = np.zeros(n_cls, dtype=np.int64)
        fp = np.zeros(n_cls, dtype=np.int64)
        fn = np.zeros(n_cls, dtype=np.int64)
        ap50 = np.zeros(n_cls, dtype=np.float64)
        ap_all = np.zeros(n_cls, dtype=np.float64)

        for (pb, pc, pconf), (gb, gc) in zip(self._predictions, self._ground_truths):
            for cls_idx in range(n_cls):
                p_mask = pc == cls_idx
                g_mask = gc == cls_idx
                p_b = pb[p_mask]
                g_b = gb[g_mask]
                cls_tp, cls_fp, cls_fn = self._match_boxes(p_b, g_b, iou_thr=0.50)
                tp[cls_idx] += cls_tp
                fp[cls_idx] += cls_fp
                fn[cls_idx] += cls_fn

        precision_arr = tp / (tp + fp + 1e-9)
        recall_arr = tp / (tp + fn + 1e-9)
        fnr_arr = fn / (fn + tp + 1e-9)

        # Compute per-class AP@0.5 and AP@0.5:0.95
        for cls_idx in range(n_cls):
            ap50[cls_idx] = self._compute_ap_at_iou(cls_idx, iou_thr=0.50)
            ap_vals = [self._compute_ap_at_iou(cls_idx, iou_thr=thr) for thr in self._iou_thresholds]
            ap_all[cls_idx] = float(np.mean(ap_vals))

        report = SafetyReport(class_names=self._class_names)
        report.precision_per_class = {
            cls: float(precision_arr[i]) for i, cls in enumerate(self._class_names)
        }
        report.recall_per_class = {
            cls: float(recall_arr[i]) for i, cls in enumerate(self._class_names)
        }
        report.false_negative_rate = {
            cls: float(fnr_arr[i]) for i, cls in enumerate(self._class_names)
        }
        report.ap50_per_class = {
            cls: float(ap50[i]) for i, cls in enumerate(self._class_names)
        }
        report.ap50_95_per_class = {
            cls: float(ap_all[i]) for i, cls in enumerate(self._class_names)
        }
        report.mAP50 = float(ap50.mean())
        report.mAP50_95 = float(ap_all.mean())
        report.mean_time_to_alert_s = (
            float(np.mean(self._tta_samples)) if self._tta_samples else None
        )

        # Safety gate
        failing = [
            cls
            for cls in _SAFETY_CRITICAL_CLASSES
            if cls in report.false_negative_rate
            and report.false_negative_rate[cls] > _FNR_THRESHOLD
        ]
        report.failing_classes = failing
        report.passes_safety_gate = len(failing) == 0

        if not report.passes_safety_gate:
            logger.warning(
                "SAFETY GATE FAILED — FNR exceeds %.2f for: %s",
                _FNR_THRESHOLD,
                failing,
            )

        return report

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._predictions.clear()
        self._ground_truths.clear()
        self._tta_samples.clear()

    # ------------------------------------------------------------------
    # Static / class-method metric functions
    # ------------------------------------------------------------------

    @staticmethod
    def precision_recall_at_iou(
        pred_boxes: np.ndarray,
        gt_boxes: np.ndarray,
        iou_thr: float = 0.50,
        pred_confs: np.ndarray | None = None,
    ) -> PRPoint:
        """
        Compute precision, recall, F1, and AP for a single class at a
        given IoU threshold.

        Parameters
        ----------
        pred_boxes: (N, 4) xyxy float32.
        gt_boxes: (M, 4) xyxy float32.
        iou_thr: IoU matching threshold.
        pred_confs: (N,) confidence scores for AP computation.

        Returns
        -------
        PRPoint
        """
        tp, fp, fn = SafetyMetrics._match_boxes(pred_boxes, gt_boxes, iou_thr)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        # Compute AP via interpolated PR curve if confidences available
        ap = 0.0
        if pred_confs is not None and len(pred_confs) > 0:
            ap = SafetyMetrics._compute_ap_from_confs(pred_boxes, gt_boxes, iou_thr, pred_confs)

        return PRPoint(
            iou_threshold=iou_thr,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            ap=float(ap),
        )

    @staticmethod
    def false_negative_rate(
        tp: int | np.ndarray,
        fn: int | np.ndarray,
    ) -> float:
        """
        Compute FNR = FN / (FN + TP).

        This is the primary safety metric: a missed detection in a
        safety-critical context can be life-threatening.

        Parameters
        ----------
        tp: True positive count (scalar or array).
        fn: False negative count (scalar or array).

        Returns
        -------
        float: FNR in [0, 1]. Lower is safer.
        """
        tp_f = float(np.sum(tp))
        fn_f = float(np.sum(fn))
        fnr = fn_f / (fn_f + tp_f + 1e-9)
        if fnr > _FNR_THRESHOLD:
            logger.warning(
                "FNR=%.4f exceeds safety gate threshold %.2f!", fnr, _FNR_THRESHOLD
            )
        return fnr

    @staticmethod
    def mean_time_to_alert(
        alert_frame_indices: list[int],
        incident_frame_indices: list[int],
        fps: float = 30.0,
    ) -> float:
        """
        Compute mean time from first incident frame to first alert emission.

        Parameters
        ----------
        alert_frame_indices:
            Frame index at which each alert was emitted.
        incident_frame_indices:
            Frame index of the first frame showing the incident.
        fps:
            Video frame rate.

        Returns
        -------
        float: Mean time-to-alert in seconds.
        """
        if not alert_frame_indices or not incident_frame_indices:
            return float("nan")

        delays = [
            (a - i) / fps
            for a, i in zip(alert_frame_indices, incident_frame_indices)
            if a >= i
        ]
        if not delays:
            return float("nan")
        mtta = float(np.mean(delays))
        logger.info(
            "Mean time-to-alert: %.3f s  (n=%d,  min=%.3f,  max=%.3f)",
            mtta,
            len(delays),
            min(delays),
            max(delays),
        )
        return mtta

    @staticmethod
    def confusion_matrix_by_severity(
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        severities: np.ndarray,
        class_names: list[str] | None = None,
        severity_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute per-severity confusion matrices.

        Parameters
        ----------
        pred_labels: (N,) predicted class index.
        gt_labels: (N,) ground-truth class index.
        severities: (N,) severity index for each sample (0=medium, 1=high, 2=critical).
        class_names: list of class name strings.
        severity_names: list of severity label strings.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are severity names; values are (n_cls x n_cls) confusion matrices.
            Also includes "overall" key for the aggregate matrix.
        """
        cls_names = class_names or ["person", "forklift", "fire", "smoke", "rack", "pallet"]
        sev_names = severity_names or ["medium", "high", "critical"]
        n_cls = len(cls_names)

        all_severities = np.unique(severities)
        cms: dict[str, np.ndarray] = {}

        for sev_idx in all_severities:
            sev_name = sev_names[int(sev_idx)] if int(sev_idx) < len(sev_names) else str(sev_idx)
            mask = severities == sev_idx
            cm = SafetyMetrics._build_confusion_matrix(
                pred_labels[mask], gt_labels[mask], n_cls
            )
            cms[sev_name] = cm

        cms["overall"] = SafetyMetrics._build_confusion_matrix(pred_labels, gt_labels, n_cls)
        return cms

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix (N, M) between two sets of xyxy boxes."""
        if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

        ax1, ay1, ax2, ay2 = (boxes_a[:, i] for i in range(4))
        bx1, by1, bx2, by2 = (boxes_b[:, i] for i in range(4))

        ix1 = np.maximum(ax1[:, None], bx1[None, :])
        iy1 = np.maximum(ay1[:, None], by1[None, :])
        ix2 = np.minimum(ax2[:, None], bx2[None, :])
        iy2 = np.minimum(ay2[:, None], by2[None, :])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / (union + 1e-9)

    @staticmethod
    def _match_boxes(
        pred: np.ndarray, gt: np.ndarray, iou_thr: float
    ) -> tuple[int, int, int]:
        """Return (TP, FP, FN) for one class, one image."""
        if gt.shape[0] == 0:
            return 0, len(pred), 0
        if pred.shape[0] == 0:
            return 0, 0, len(gt)

        iou = SafetyMetrics._box_iou(pred, gt)
        matched_gt: set[int] = set()
        tp = 0
        for p_idx in range(len(pred)):
            best_j = int(np.argmax(iou[p_idx]))
            if iou[p_idx, best_j] >= iou_thr and best_j not in matched_gt:
                tp += 1
                matched_gt.add(best_j)
        fp = len(pred) - tp
        fn = len(gt) - len(matched_gt)
        return tp, fp, fn

    @staticmethod
    def _compute_ap_from_confs(
        pred_boxes: np.ndarray,
        gt_boxes: np.ndarray,
        iou_thr: float,
        pred_confs: np.ndarray,
    ) -> float:
        """Compute AP via 101-point interpolation of the PR curve."""
        sort_idx = np.argsort(-pred_confs)
        sorted_boxes = pred_boxes[sort_idx]
        matched_gt: set[int] = set()
        iou_matrix = SafetyMetrics._box_iou(sorted_boxes, gt_boxes)

        tps = []
        fps = []
        for p_idx in range(len(sorted_boxes)):
            if gt_boxes.shape[0] == 0:
                fps.append(1)
                tps.append(0)
                continue
            best_j = int(np.argmax(iou_matrix[p_idx]))
            if iou_matrix[p_idx, best_j] >= iou_thr and best_j not in matched_gt:
                tps.append(1)
                fps.append(0)
                matched_gt.add(best_j)
            else:
                tps.append(0)
                fps.append(1)

        tp_cum = np.cumsum(tps).astype(float)
        fp_cum = np.cumsum(fps).astype(float)
        n_gt = max(len(gt_boxes), 1)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall = tp_cum / n_gt

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p_at_r = precision[recall >= t]
            ap += float(p_at_r.max()) if p_at_r.size > 0 else 0.0
        return ap / 101.0

    def _compute_ap_at_iou(self, cls_idx: int, iou_thr: float) -> float:
        """Compute AP for one class at one IoU threshold from accumulated buffers."""
        all_preds: list[tuple[np.ndarray, np.ndarray]] = []
        all_gts: list[np.ndarray] = []
        for (pb, pc, pconf), (gb, gc) in zip(self._predictions, self._ground_truths):
            p_mask = pc == cls_idx
            g_mask = gc == cls_idx
            all_preds.append((pb[p_mask], pconf[p_mask]))
            all_gts.append(gb[g_mask])

        # Concatenate across images
        if not all_preds:
            return 0.0
        all_pb = np.concatenate([p[0] for p in all_preds], axis=0)
        all_pc = np.concatenate([p[1] for p in all_preds], axis=0)
        all_gb = np.concatenate(all_gts, axis=0)
        if len(all_pb) == 0 or len(all_gb) == 0:
            return 0.0
        return self._compute_ap_from_confs(all_pb, all_gb, iou_thr, all_pc)

    @staticmethod
    def _build_confusion_matrix(
        preds: np.ndarray, gts: np.ndarray, n_cls: int
    ) -> np.ndarray:
        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        for p, g in zip(preds.astype(int), gts.astype(int)):
            if 0 <= g < n_cls and 0 <= p < n_cls:
                cm[g, p] += 1
        return cm

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_confusion_matrix(
        cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix"
    ) -> None:
        """Pretty-print a confusion matrix to the logger."""
        header = f"{'':>14}" + " ".join(f"{n:>10}" for n in class_names)
        lines = [title, header]
        for i, row_name in enumerate(class_names):
            row = f"{row_name:>14}" + " ".join(f"{cm[i, j]:>10}" for j in range(len(class_names)))
            lines.append(row)
        logger.info("\n".join(lines))

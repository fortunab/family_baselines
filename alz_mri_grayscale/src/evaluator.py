"""Clinical Classification Performance Evaluator for Alzheimer MRI Benchmark.

Calculates 10 clinical metrics:
- Overall Accuracy & Balanced Accuracy
- Macro & Weighted Precision
- Macro & Weighted Recall (Sensitivity)
- Macro Specificity
- Macro & Weighted F1-Score
- Cohen's Kappa & Matthews Correlation Coefficient (MCC)
- ROC-AUC Macro & Multiclass Log-Loss
- Confusion Matrix

Validates user condition: Verifies whether metrics exceed the >80% (0.80) clinical benchmark threshold.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_specificity_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calculates macro-averaged specificity across all 4 Alzheimer stages."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        denom = tn + fp
        spec = float(tn / denom) if denom > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))


def evaluate_clinical_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    threshold_target: float = 0.80,
) -> Dict[str, Any]:
    """Computes comprehensive clinical metrics and verifies if key metrics exceed the >80% threshold."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is not None:
        num_classes = max(num_classes, len(class_names))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    spec_macro = calculate_specificity_macro(y_true, y_pred, num_classes)
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # ROC-AUC & Log Loss
    roc_auc = None
    loss_val = None
    if y_prob is not None:
        try:
            if y_prob.shape[1] == num_classes and len(np.unique(y_true)) > 1:
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                loss_val = float(log_loss(y_true, y_prob))
        except Exception:
            roc_auc = None
            loss_val = None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()

    # Per-class breakdown
    per_class = {}
    for i in range(num_classes):
        c_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        c_true = (y_true == i).astype(int)
        c_pred = (y_pred == i).astype(int)
        per_class[c_name] = {
            "precision": float(precision_score(c_true, c_pred, zero_division=0)),
            "recall": float(recall_score(c_true, c_pred, zero_division=0)),
            "f1_score": float(f1_score(c_true, c_pred, zero_division=0)),
            "support": int(np.sum(y_true == i)),
        }

    # Verify >80% threshold target
    threshold_checks = {
        "accuracy_gt_80": acc >= threshold_target,
        "balanced_accuracy_gt_80": bal_acc >= threshold_target,
        "precision_macro_gt_80": prec_macro >= threshold_target,
        "recall_macro_gt_80": rec_macro >= threshold_target,
        "specificity_macro_gt_80": spec_macro >= threshold_target,
        "f1_macro_gt_80": f1_mac >= threshold_target,
        "f1_weighted_gt_80": f1_wt >= threshold_target,
    }
    target_exceeded = acc >= threshold_target and f1_wt >= threshold_target and spec_macro >= threshold_target

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "specificity_macro": round(spec_macro, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "cohen_kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
        "roc_auc_macro": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "log_loss": round(loss_val, 4) if loss_val is not None else "N/A",
        "threshold_target": threshold_target,
        "threshold_checks": threshold_checks,
        "target_met_gt_80_percent": target_exceeded,
        "confusion_matrix": cm,
        "per_class_metrics": per_class,
    }

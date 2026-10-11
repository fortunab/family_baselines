"""Clinical Metric Evaluator for Diabetic Retinopathy and Medical ML.

Calculates:
1. Accuracy & Balanced Accuracy
2. Quadratic Weighted Kappa (QWK) - The clinical standard for ordinal DR grading
3. Multi-Class Macro Sensitivity (Recall), Specificity, and Precision
4. Macro F1 & Weighted F1
5. Binary Referable DR Metrics (Sensitivity, Specificity, PPV, NPV at Grade >= 2)
6. Multi-Class AUROC (One-vs-Rest)
7. Expected Calibration Error (ECE)
8. Domain Shift Generalization Gap (Intra-DB vs Cross-DB transfer)
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Quadratic Weighted Kappa (QWK) for ordinal clinical grading."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Computes 10-bin Expected Calibration Error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    bins = np.linspace(0, 1, num_bins + 1)
    total_samples = len(y_true)

    for b in range(num_bins):
        bin_idx = (confidences >= bins[b]) & (confidences < bins[b + 1])
        if np.any(bin_idx):
            bin_conf = np.mean(confidences[bin_idx])
            bin_acc = np.mean(accuracies[bin_idx])
            bin_weight = np.sum(bin_idx) / total_samples
            ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_clinical_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    referral_threshold: int = 2,
) -> Dict[str, Any]:
    """Computes full suite of clinical and statistical metrics for Diabetic Retinopathy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    num_classes = len(class_names) if class_names else len(np.unique(y_t))

    acc = accuracy_score(y_t, y_p)
    bal_acc = balanced_accuracy_score(y_t, y_p)
    qwk = compute_quadratic_weighted_kappa(y_t, y_p)

    macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_t, y_p, average="weighted", zero_division=0)
    macro_prec = precision_score(y_t, y_p, average="macro", zero_division=0)
    macro_rec = recall_score(y_t, y_p, average="macro", zero_division=0)

    # Specificity per class & macro specificity
    cm = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))
    specificities = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = float(np.mean(specificities))

    # Binary Referable DR Task (No/Mild DR vs Referable DR Grade >= 2)
    bin_true = (y_t >= referral_threshold).astype(int)
    bin_pred = (y_p >= referral_threshold).astype(int)
    bin_cm = confusion_matrix(bin_true, bin_pred, labels=[0, 1])
    btn, bfp, bfn, btp = bin_cm.ravel()

    referral_sensitivity = btp / (btp + bfn) if (btp + bfn) > 0 else 0.0
    referral_specificity = btn / (btn + bfp) if (btn + bfp) > 0 else 0.0
    referral_ppv = btp / (btp + bfp) if (btp + bfp) > 0 else 0.0
    referral_npv = btn / (btn + bfn) if (btn + bfn) > 0 else 0.0

    # AUROC & Calibration
    auroc = 0.5
    ece = 0.0
    if y_prob is not None:
        try:
            auroc = float(roc_auc_score(y_t, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            auroc = 0.5
        ece = compute_expected_calibration_error(y_t, y_prob)

    metrics = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_sensitivity_recall": round(float(macro_rec), 4),
        "macro_specificity": round(float(macro_spec), 4),
        "auroc_ovr": round(float(auroc), 4),
        "expected_calibration_error": round(float(ece), 4),
        "binary_referral": {
            "referral_sensitivity": round(float(referral_sensitivity), 4),
            "referral_specificity": round(float(referral_specificity), 4),
            "referral_ppv": round(float(referral_ppv), 4),
            "referral_npv": round(float(referral_npv), 4),
        },
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def compute_domain_shift_gap(
    intra_metrics: Dict[str, Any],
    cross_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Computes transfer degradation gap: Delta = Intra_DB - Cross_DB."""
    return {
        "delta_accuracy": round(float(intra_metrics["accuracy"] - cross_metrics["accuracy"]), 4),
        "delta_qwk": round(
            float(intra_metrics["quadratic_weighted_kappa"] - cross_metrics["quadratic_weighted_kappa"]), 4
        ),
        "delta_macro_f1": round(float(intra_metrics["macro_f1"] - cross_metrics["macro_f1"]), 4),
        "delta_referral_sensitivity": round(
            float(
                intra_metrics["binary_referral"]["referral_sensitivity"]
                - cross_metrics["binary_referral"]["referral_sensitivity"]
            ),
            4,
        ),
    }

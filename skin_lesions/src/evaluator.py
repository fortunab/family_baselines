"""Comprehensive Clinical Dermatology Evaluator for 7-Class Skin Lesions.

Calculates:
1. Multi-Class: Accuracy, Balanced Accuracy, Cohen's Kappa, Macro Precision, Recall, F1
2. Calibration: Expected Calibration Error (ECE)
3. Clinical Malignancy Referral (Melanoma + BCC + AKIEC vs. Benign):
   - Referral Sensitivity (Recall on Malignancy)
   - Referral Specificity
   - Positive Predictive Value (PPV)
   - Negative Predictive Value (NPV)
4. Multi-Class Confusion Matrix
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

from src.dataset_generator import CLASS_NAMES, MALIGNANT_CLASSES


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes multiclass Expected Calibration Error (ECE) based on maximum confidence."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_clinical_skin_lesions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: List[str] = None,
    malignant_classes: List[str] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class and clinical referral metrics."""
    if class_names is None:
        class_names = CLASS_NAMES
    if malignant_classes is None:
        malignant_classes = MALIGNANT_CLASSES

    num_classes = len(class_names)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class specificity
    spec_list = []
    for c in range(num_classes):
        tn = np.sum(cm) - (np.sum(cm[c, :]) + np.sum(cm[:, c]) - cm[c, c])
        fp = np.sum(cm[:, c]) - cm[c, c]
        spec_c = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        spec_list.append(spec_c)
    macro_spec = float(np.mean(spec_list))

    # AUROC OvR
    auroc_val = 0.5
    if y_prob is not None:
        try:
            # Check if all classes present in y_true
            if len(np.unique(y_true)) == num_classes:
                auroc_val = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            else:
                # Present classes only
                present_classes = np.unique(y_true)
                auroc_val = float(
                    roc_auc_score(
                        y_true,
                        y_prob[:, present_classes],
                        multi_class="ovr",
                        average="macro",
                        labels=present_classes,
                    )
                )
        except Exception:
            auroc_val = 0.5

    # ECE
    ece_val = 0.0
    if y_prob is not None:
        ece_val = compute_expected_calibration_error(y_true, y_prob)

    # Binary Clinical Malignancy Referral (MEL, BCC, AKIEC = 1 vs Benign = 0)
    mal_indices = [class_names.index(c) for c in malignant_classes if c in class_names]
    y_true_binary = np.isin(y_true, mal_indices).astype(int)
    y_pred_binary = np.isin(y_pred, mal_indices).astype(int)

    tp_bin = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    fp_bin = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    tn_bin = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fn_bin = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

    ref_sens = float(tp_bin / (tp_bin + fn_bin)) if (tp_bin + fn_bin) > 0 else 0.0
    ref_spec = float(tn_bin / (tn_bin + fp_bin)) if (tn_bin + fp_bin) > 0 else 0.0
    ref_ppv = float(tp_bin / (tp_bin + fp_bin)) if (tp_bin + fp_bin) > 0 else 0.0
    ref_npv = float(tn_bin / (tn_bin + fn_bin)) if (tn_bin + fn_bin) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_specificity": round(float(macro_spec), 4),
        "auroc_ovr": round(float(auroc_val), 4),
        "expected_calibration_error": round(float(ece_val), 4),
        "binary_malignancy_referral": {
            "referral_sensitivity": round(ref_sens, 4),
            "referral_specificity": round(ref_spec, 4),
            "referral_ppv": round(ref_ppv, 4),
            "referral_npv": round(ref_npv, 4),
            "tp": tp_bin,
            "fp": fp_bin,
            "tn": tn_bin,
            "fn": fn_bin,
        },
        "confusion_matrix": cm.tolist(),
        "per_class_specificity": [round(float(s), 4) for s in spec_list],
    }

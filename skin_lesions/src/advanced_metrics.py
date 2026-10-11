"""Advanced Clinical & Fairness Metrics for 7-Class Skin Lesion Classification.

Implements:
1. Wilson 95% Confidence Intervals for Malignancy Referral Sensitivity, Specificity, PPV, NPV
2. Multiclass Brier Score across all 7 ISIC/HAM10000 classes
3. Partial Area Under ROC Curve (pAUC) at high specificity (Specificity >= 80%, FPR <= 20%)
4. Fitzpatrick Skin Tone Fairness & Disparity Analysis (Skin Phototypes I to VI)
5. Decision Curve Analysis (DCA): Clinical Net Benefit curves vs. "Biopsy All" vs. "Biopsy None"
6. Critical Melanoma (MEL) and Basal Cell Carcinoma (BCC) Emergency Referral Recall
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import auc, roc_curve

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT_CLASSES = ["MEL", "BCC", "AKIEC"]


def compute_wilson_score_interval(
    count: int,
    nobs: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Computes two-sided Wilson score confidence interval for a binomial proportion."""
    if nobs == 0:
        return 0.0, 0.0

    z_dict = {0.90: 1.644853, 0.95: 1.959964, 0.99: 2.575829}
    z = z_dict.get(confidence, 1.959964)

    p_hat = count / nobs
    denominator = 1.0 + (z**2) / nobs
    centre = (p_hat + (z**2) / (2.0 * nobs)) / denominator
    spread = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / nobs) + ((z**2) / (4.0 * (nobs**2))))

    lower = max(0.0, float(centre - spread))
    upper = min(1.0, float(centre + spread))
    return round(lower, 4), round(upper, 4)


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes multiclass Brier score: 1/N * sum_i sum_c (p_ic - y_ic)^2."""
    n_samples, n_classes = y_prob.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i, y in enumerate(y_true):
        if 0 <= y < n_classes:
            one_hot[i, y] = 1.0
    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def compute_partial_auc(
    y_true_binary: np.ndarray,
    y_prob_positive: np.ndarray,
    min_specificity: float = 0.80,
) -> float:
    """Computes partial Area Under ROC Curve (pAUC) for specificity >= min_specificity.

    In dermatology, high specificity is critical to avoid overburdening pathology with negative biopsies.
    Normalized to range [0, 1] over the region FPR in [0, max_fpr].
    """
    max_fpr = 1.0 - min_specificity
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_positive)

    # Filter to region fpr <= max_fpr
    valid_idx = fpr <= max_fpr
    sub_fpr = fpr[valid_idx]
    sub_tpr = tpr[valid_idx]

    # Interpolate at max_fpr if necessary
    if len(sub_fpr) > 0 and sub_fpr[-1] < max_fpr:
        last_tpr = np.interp(max_fpr, fpr, tpr)
        sub_fpr = np.append(sub_fpr, max_fpr)
        sub_tpr = np.append(sub_tpr, last_tpr)

    raw_pauc = auc(sub_fpr, sub_tpr)
    # Normalized pAUC: raw_pauc / max_fpr
    normalized_pauc = raw_pauc / max_fpr if max_fpr > 0 else 0.5
    return round(float(normalized_pauc), 4)


def compute_decision_curve_analysis(
    y_true_binary: np.ndarray,
    y_prob_positive: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes Decision Curve Analysis (DCA) Net Benefit across clinical threshold probabilities.

    Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    Compares Model vs. 'Biopsy All' vs. 'Biopsy None'.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.70, 27)

    n_total = len(y_true_binary)
    prevalence = np.mean(y_true_binary)

    model_net_benefit: List[float] = []
    biopsy_all_net_benefit: List[float] = []
    biopsy_none_net_benefit: List[float] = [0.0] * len(thresholds)

    for p_t in thresholds:
        weight = p_t / (1.0 - p_t)

        # Model decisions at threshold p_t
        y_pred = (y_prob_positive >= p_t).astype(int)
        tp = np.sum((y_true_binary == 1) & (y_pred == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred == 1))

        nb_model = (tp / n_total) - (fp / n_total) * weight
        model_net_benefit.append(float(nb_model))

        # 'Biopsy All' strategy: all patients are positive
        tp_all = np.sum(y_true_binary == 1)
        fp_all = np.sum(y_true_binary == 0)
        nb_all = (tp_all / n_total) - (fp_all / n_total) * weight
        biopsy_all_net_benefit.append(float(nb_all))

    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "model_net_benefit": [round(nb, 4) for nb in model_net_benefit],
        "biopsy_all_net_benefit": [round(nb, 4) for nb in biopsy_all_net_benefit],
        "biopsy_none_net_benefit": biopsy_none_net_benefit,
        "disease_prevalence": round(float(prevalence), 4),
    }


def compute_fitzpatrick_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fitzpatrick_types: List[str],
    malignant_classes: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes fairness and disparity metrics across Fitzpatrick phototype groups.

    Standard Dermatological Grouping:
    - Light Skin: Types I-II
    - Medium Skin: Types III-IV
    - Dark Skin: Types V-VI
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if malignant_classes is None:
        malignant_classes = MALIGNANT_CLASSES

    mal_indices = [class_names.index(c) for c in malignant_classes if c in class_names]
    y_true_bin = np.isin(y_true, mal_indices).astype(int)
    y_pred_bin = np.isin(y_pred, mal_indices).astype(int)

    group_mapping = {
        "Type_I": "Light_I_II",
        "Type_II": "Light_I_II",
        "Type_III": "Medium_III_IV",
        "Type_IV": "Medium_III_IV",
        "Type_V": "Dark_V_VI",
        "Type_VI": "Dark_V_VI",
    }

    subgroups = ["Light_I_II", "Medium_III_IV", "Dark_V_VI"]
    group_results: Dict[str, Any] = {}
    sensitivities: List[float] = []

    for grp in subgroups:
        mask = np.array([group_mapping.get(t, "Medium_III_IV") == grp for t in fitzpatrick_types])
        n_grp = int(np.sum(mask))

        if n_grp == 0:
            group_results[grp] = {"n_samples": 0, "sensitivity": 0.0, "specificity": 0.0}
            continue

        y_true_g = y_true_bin[mask]
        y_pred_g = y_pred_bin[mask]

        tp = int(np.sum((y_true_g == 1) & (y_pred_g == 1)))
        fn = int(np.sum((y_true_g == 1) & (y_pred_g == 0)))
        tn = int(np.sum((y_true_g == 0) & (y_pred_g == 0)))
        fp = int(np.sum((y_true_g == 0) & (y_pred_g == 1)))

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        group_results[grp] = {
            "n_samples": n_grp,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
        }
        if (tp + fn) > 0:
            sensitivities.append(sens)

    # Disparity calculations
    if len(sensitivities) >= 2:
        min_sens = min(sensitivities)
        max_sens = max(sensitivities)
        disparity_ratio = float(min_sens / max_sens) if max_sens > 0 else 1.0
        max_disparity = float(max_sens - min_sens)
    else:
        disparity_ratio = 1.0
        max_disparity = 0.0

    return {
        "groups": group_results,
        "disparity_ratio": round(disparity_ratio, 4),
        "max_disparity": round(max_disparity, 4),
        "fairness_compliant": bool(disparity_ratio >= 0.80),  # 80% four-fifths fairness rule
    }


def compute_critical_cancer_recalls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes exact recall for high-priority malignant entities: Melanoma and BCC."""
    if class_names is None:
        class_names = CLASS_NAMES

    recalls: Dict[str, Any] = {}
    for cancer in ["MEL", "BCC"]:
        if cancer in class_names:
            c_idx = class_names.index(cancer)
            tp = int(np.sum((y_true == c_idx) & (y_pred == c_idx)))
            total = int(np.sum(y_true == c_idx))
            rec = float(tp / total) if total > 0 else 0.0
            recalls[f"{cancer.lower()}_recall"] = round(rec, 4)
            recalls[f"{cancer.lower()}_tp"] = tp
            recalls[f"{cancer.lower()}_total"] = total
    return recalls

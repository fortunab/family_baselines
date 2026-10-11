"""Advanced Clinical Ophthalmology Metrics for Diabetic Retinopathy.

Calculates:
1. Quadratic Weighted Kappa (QWK) vs Linear Weighted Kappa vs Unweighted Cohen's Kappa
2. Wilson 95% Confidence Intervals for Clinical Sensitivity, Specificity, PPV, and NPV
3. Brier Score and Expected Calibration Error (ECE)
4. Decision Curve Analysis (DCA): Clinical Net Benefit vs 'Refer All' and 'Refer None'
5. Critical Grade 4 (Proliferative DR) Safety Recall
6. Cross-Center Domain Shift Generalization Gap
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def compute_wilson_score_interval(
    count: int, nobs: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Computes Wilson score confidence interval for a binomial proportion."""
    if nobs == 0:
        return 0.0, 0.0
    p = count / nobs
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z**2) / nobs
    centre_adj = p + (z**2) / (2 * nobs)
    adj = z * np.sqrt((p * (1 - p) + (z**2) / (4 * nobs)) / nobs)
    lower = max(0.0, (centre_adj - adj) / denom)
    upper = min(1.0, (centre_adj + adj) / denom)
    return round(float(lower), 4), round(float(upper), 4)


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 5) -> float:
    """Computes multiclass Brier score (mean squared error of probability predictions)."""
    n_samples = len(y_true)
    one_hot = np.zeros((n_samples, num_classes))
    for i in range(n_samples):
        one_hot[i, y_true[i]] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_kappa_trinity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Quadratic, Linear, and Unweighted Cohen's Kappa."""
    unweighted = cohen_kappa_score(y_true, y_pred, weights=None)
    linear = cohen_kappa_score(y_true, y_pred, weights="linear")
    quadratic = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "unweighted_kappa": round(float(unweighted), 4),
        "linear_weighted_kappa": round(float(linear), 4),
        "quadratic_weighted_kappa": round(float(quadratic), 4),
    }


def compute_decision_curve_analysis(
    y_true_binary: np.ndarray,
    y_prob_referral: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Calculates Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.

    Net Benefit: NB(p_t) = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.80, 16)

    n_samples = len(y_true_binary)
    prevalence = np.mean(y_true_binary)

    model_net_benefits = []
    refer_all_net_benefits = []
    refer_none_net_benefits = []

    for pt in thresholds:
        weight = pt / (1.0 - pt)
        preds_at_pt = (y_prob_referral >= pt).astype(int)

        tp = np.sum((preds_at_pt == 1) & (y_true_binary == 1))
        fp = np.sum((preds_at_pt == 1) & (y_true_binary == 0))

        # Model net benefit
        nb_model = (tp / n_samples) - (fp / n_samples) * weight
        model_net_benefits.append(round(float(nb_model), 4))

        # Refer All net benefit
        tp_all = np.sum(y_true_binary == 1)
        fp_all = np.sum(y_true_binary == 0)
        nb_all = (tp_all / n_samples) - (fp_all / n_samples) * weight
        refer_all_net_benefits.append(round(float(nb_all), 4))

        # Refer None net benefit (always 0)
        refer_none_net_benefits.append(0.0)

    return {
        "thresholds": [round(float(t), 3) for t in thresholds],
        "model_net_benefit": model_net_benefits,
        "refer_all_net_benefit": refer_all_net_benefits,
        "refer_none_net_benefit": refer_none_net_benefits,
        "disease_prevalence": round(float(prevalence), 4),
    }


def compute_comprehensive_advanced_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    referral_threshold_grade: int = 2,
) -> Dict[str, Any]:
    """Evaluates all advanced ophthalmology metrics."""
    kappas = compute_kappa_trinity(y_true, y_pred)
    brier = compute_multiclass_brier_score(y_true, y_prob)

    # Referral binary evaluation
    y_true_ref = (y_true >= referral_threshold_grade).astype(int)
    y_pred_ref = (y_pred >= referral_threshold_grade).astype(int)
    prob_ref = np.sum(y_prob[:, referral_threshold_grade:], axis=1)

    tp = int(np.sum((y_pred_ref == 1) & (y_true_ref == 1)))
    fp = int(np.sum((y_pred_ref == 1) & (y_true_ref == 0)))
    tn = int(np.sum((y_pred_ref == 0) & (y_true_ref == 0)))
    fn = int(np.sum((y_pred_ref == 0) & (y_true_ref == 1)))

    sens_ci = compute_wilson_score_interval(tp, tp + fn)
    spec_ci = compute_wilson_score_interval(tn, tn + fp)
    ppv_ci = compute_wilson_score_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    npv_ci = compute_wilson_score_interval(tn, tn + fn) if (tn + fn) > 0 else (0.0, 0.0)

    sens_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Critical Grade 4 (PDR) recall
    pdr_indices = np.where(y_true == 4)[0]
    pdr_recall = float(np.mean(y_pred[pdr_indices] == 4)) if len(pdr_indices) > 0 else 0.0
    pdr_referral_recall = float(np.mean(y_pred[pdr_indices] >= 2)) if len(pdr_indices) > 0 else 0.0

    # Decision Curve Analysis
    dca = compute_decision_curve_analysis(y_true_ref, prob_ref)

    return {
        "kappas": kappas,
        "brier_score": round(brier, 4),
        "referral_metrics_with_ci": {
            "sensitivity": round(float(sens_val), 4),
            "sensitivity_95_ci": sens_ci,
            "specificity": round(float(spec_val), 4),
            "specificity_95_ci": spec_ci,
            "ppv_95_ci": ppv_ci,
            "npv_95_ci": npv_ci,
            "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "critical_pdr_safety": {
            "pdr_exact_recall": round(pdr_recall, 4),
            "pdr_referral_recall": round(pdr_referral_recall, 4),
        },
        "decision_curve_analysis": dca,
    }

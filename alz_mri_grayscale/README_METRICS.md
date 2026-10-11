# Clinical Metrics & >80% Threshold Validation Guide

Mathematical specifications and clinical definitions for the 10 classification metrics evaluated in the **Alzheimer MRI Grayscale Benchmark Suite**.

---

## 1. Clinical Evaluation Metrics

| Metric Name | Mathematical Definition | Clinical Interpretation | Target Threshold |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{\sum_{i=1}^4 TP_i}{N}$ | Overall proportion of correct dementia stage diagnoses. | **> 80% (0.80)** |
| **Balanced Accuracy** | $\frac{1}{4}\sum_{i=1}^4 \frac{TP_i}{TP_i + FN_i}$ | Accuracy adjusted for class imbalance (fewer Moderate cases). | **> 80% (0.80)** |
| **Macro Precision** | $\frac{1}{4}\sum_{i=1}^4 \frac{TP_i}{TP_i + FP_i}$ | Confidence that a predicted dementia stage is correct. | **> 80% (0.80)** |
| **Weighted Precision** | $\sum_{i=1}^4 w_i \frac{TP_i}{TP_i + FP_i}$ | Precision weighted by prevalence in cohort. | **> 80% (0.80)** |
| **Macro Recall (Sensitivity)** | $\frac{1}{4}\sum_{i=1}^4 \frac{TP_i}{TP_i + FN_i}$ | Sensitivity to neurodegenerative changes without false negatives. | **> 80% (0.80)** |
| **Macro Specificity** | $\frac{1}{4}\sum_{i=1}^4 \frac{TN_i}{TN_i + FP_i}$ | True negative rate correctly excluding non-matching stages. | **> 80% (0.80)** |
| **Macro F1-Score** | $2 \cdot \frac{\text{Prec}_{\text{macro}} \cdot \text{Rec}_{\text{macro}}}{\text{Prec}_{\text{macro}} + \text{Rec}_{\text{macro}}}$ | Harmonic balance between precision and sensitivity. | **> 80% (0.80)** |
| **Weighted F1-Score** | $\sum_{i=1}^4 w_i F1_i$ | Overall prevalence-adjusted F1 performance. | **> 80% (0.80)** |
| **Cohen's Kappa ($\kappa$)** | $\frac{p_o - p_e}{1 - p_e}$ | Diagnostic agreement penalizing chance agreement. | High stringency |
| **Matthews Correlation (MCC)** | Multiclass Covariance Formula | Balanced correlation between actual and predicted diagnoses. | High stringency |
| **ROC-AUC Macro (OVR)** | One-vs-Rest AUC averaged across stages | Threshold-independent discriminability metric. | **> 80% (0.80)** |
| **Multiclass Log-Loss** | $-\frac{1}{N}\sum_{n=1}^N \sum_{c=1}^4 y_{n,c} \ln(p_{n,c})$ | Cross-entropy penalizing misclassifications. | Lower is better |

---

## 2. Threshold Verification Logic
The evaluator module (`src/evaluator.py`) verifies each metric against the $0.80$ threshold:
```python
threshold_checks = {
    "accuracy_gt_80": acc >= 0.80,
    "balanced_accuracy_gt_80": bal_acc >= 0.80,
    "precision_macro_gt_80": prec_macro >= 0.80,
    "recall_macro_gt_80": rec_macro >= 0.80,
    "specificity_macro_gt_80": spec_macro >= 0.80,
    "f1_macro_gt_80": f1_mac >= 0.80,
    "f1_weighted_gt_80": f1_wt >= 0.80,
}
```
All models achieving $\ge 80\%$ on Accuracy, Weighted F1, and Specificity are flagged as `[PASSED >80%]`.

# Clinical Evaluation Metrics & Quadratic Weighted Kappa

In medical diagnostics, general machine learning metrics (such as unweighted accuracy) can be misleading. This guide explains why specific clinical metrics are used for diabetic retinopathy grading.

---

## 1. Quadratic Weighted Kappa (QWK) — The Clinical Standard

Standard accuracy treats all classification mistakes identically. However, in Diabetic Retinopathy:
- Confusing **Grade 0 (No DR)** with **Grade 1 (Mild NPDR)** is a minor discrepancy (both are non-referable).
- Confusing **Grade 0 (No DR)** with **Grade 4 (Proliferative DR)** is a **catastrophic clinical failure** (a patient with impending blindness is sent home).

**Quadratic Weighted Kappa** ($\kappa_w$) penalizes errors proportionally to the square of their distance on the ordinal clinical scale:

$$w_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$

$$\kappa_w = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}$$

where:
- $O_{ij}$ is the observed confusion matrix between grade $i$ and grade $j$.
- $E_{ij}$ is the expected confusion matrix under random chance agreement:
  $$E_{ij} = \frac{\sum_k O_{ik} \sum_k O_{kj}}{N}$$

### Clinical Interpretation of $\kappa_w$:
- $< 0.40$: Poor agreement
- $0.40 - 0.60$: Moderate clinical agreement
- $0.60 - 0.80$: Substantial clinical agreement
- **$> 0.80$**: Almost perfect agreement (Clinical Grade Standard)

---

## 2. Binary Referable DR Metrics

The clinical referral decision determines whether a patient requires specialist ophthalmologist intervention:
- **Non-Referable DR**: Grade 0 (No DR) & Grade 1 (Mild NPDR)
- **Referable DR**: Grade 2 (Moderate), Grade 3 (Severe), Grade 4 (Proliferative)

We evaluate:
- **Referral Sensitivity (Recall)**: $TP / (TP + FN)$ — Measures how many patients needing specialist referral are correctly identified.
- **Referral Specificity**: $TN / (TN + FP)$ — Measures how many healthy patients are spared unnecessary specialist referrals.
- **Positive Predictive Value (PPV)**: Precision of a referral alert.
- **Negative Predictive Value (NPV)**: Confidence that a patient cleared for annual screening truly does not have advanced pathology.

---

## 3. Expected Calibration Error (ECE)

A clinical AI model must know when it is uncertain. **ECE** measures how well predicted confidence probabilities align with actual empirical accuracy across 10 confidence bins:

$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

A low ECE ($\le 0.08$) guarantees that when the model outputs $90\%$ confidence, it is correct approximately $90\%$ of the time.

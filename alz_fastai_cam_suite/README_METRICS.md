# Multi-Step Evaluation Metrics Guide: Alzheimer MRI fastai CAM Suite

The suite evaluates metrics at **every step** of the clinical pipeline:
1. **Step 1: Clinical Model Classification Step Metrics** (10 quantitative metrics)
2. **Step 2: CAM Explainability & Attribution Step Metrics** (5 faithfulness and localization metrics)

---

## 1. Step 1: Clinical Classification Performance Metrics

For 4-class Alzheimer MRI diagnosis ($C = 4$: NonDemented, VeryMildDemented, MildDemented, ModerateDemented):

| Metric Name | Mathematical Definition | Clinical / Diagnostic Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{\sum_{i=1}^C TP_i}{N}$ | Overall diagnostic correctness across all scan slices. |
| **Balanced Accuracy** | $\frac{1}{C}\sum_{i=1}^C \frac{TP_i}{TP_i + FN_i}$ | Mitigates class imbalance (e.g. fewer ModerateDemented slices). |
| **Macro Precision** | $\frac{1}{C}\sum_{i=1}^C \frac{TP_i}{TP_i + FP_i}$ | Probability that a predicted stage of dementia is truly correct. |
| **Weighted Precision** | $\sum_{i=1}^C w_i \frac{TP_i}{TP_i + FP_i}$ | Precision weighted by class prevalence in the cohort. |
| **Macro Recall (Sensitivity)** | $\frac{1}{C}\sum_{i=1}^C \frac{TP_i}{TP_i + FN_i}$ | Capacity to detect neurodegenerative pathology without missing patients. |
| **Macro Specificity** | $\frac{1}{C}\sum_{i=1}^C \frac{TN_i}{TN_i + FP_i}$ | Ability to correctly identify patients who do NOT have that specific dementia stage. |
| **Macro F1-Score** | $2 \cdot \frac{\text{Prec}_{\text{macro}} \cdot \text{Rec}_{\text{macro}}}{\text{Prec}_{\text{macro}} + \text{Rec}_{\text{macro}}}$ | Harmonic balance between precision and recall across all stages. |
| **Weighted F1-Score** | $\sum_{i=1}^C w_i F1_i$ | Prevalence-adjusted overall F1 performance. |
| **Cohen's Kappa ($\kappa$)** | $\frac{p_o - p_e}{1 - p_e}$ | Inter-rater agreement between model and radiologist, penalizing chance agreement. |
| **Matthews Correlation (MCC)** | Multiclass Covariance Formula | Balanced metric robust to skewness; ranges from -1 to +1. |
| **ROC-AUC (Macro OVR)** | One-vs-Rest AUC averaged across classes | Diagnostic discriminability threshold-independent metric. |
| **Multiclass Log-Loss** | $-\frac{1}{N}\sum_{n=1}^N \sum_{c=1}^C y_{n,c} \ln(p_{n,c})$ | Cross-entropy penalizing high-confidence misclassifications. |

---

## 2. Step 2: CAM Explainability & Attribution Metrics

Evaluates how faithfully and accurately the heatmap attributions reflect actual model decisions and anatomical tissue:

### 1. Average Drop (%)
Measures the degradation in target class confidence when uninformative/background regions are zeroed out by the CAM heatmap:

$$\text{Average Drop (\%)} = \frac{\max(0, Y_c - O_c)}{Y_c} \times 100$$

where:
- $Y_c$ is the model output score on the original brain MRI slice $X$.
- $O_c$ is the model output score on the masked input slice $X \odot M_c$.
- A lower drop indicates that the CAM successfully preserved the model's evidentiary basis.

### 2. Ablation Drop (%)
Measures how much the model score drops when the salient region identified by the CAM is removed ($X \odot (1 - M_c)$):
$$\text{Ablation Drop (\%)} = \frac{\max(0, Y_c - O_{c,\text{ablated}})}{Y_c} \times 100$$
A high ablation drop proves that the removed tissue was indeed essential for the diagnosis.

### 3. Increase in Confidence (%)
The percentage of instances where model prediction probability increases when focused purely on the CAM-identified region:
$$\text{Increase in Confidence (\%)} = \mathbb{I}(O_c > Y_c) \times 100$$
Demonstrates removal of background noise leading to higher diagnostic certainty.

### 4. Brain Energy Localization Ratio
In MRI scans, background pixels (air, skull coil padding) have negligible intensity. This metric calculates the fraction of attribution energy localized within the brain tissue parenchyma:

$$\text{Energy Ratio} = \frac{\sum_{(i,j) \in \text{BrainMask}} M_c(i,j)}{\sum_{(i,j)} M_c(i,j) + \epsilon}$$

An energy ratio $> 0.90$ confirms the attribution map is not hallucinating onto imaging background or scan margins.

### 5. Attribution Shannon Entropy
Quantifies the concentration vs dispersion of attention over the 2D slice:

$$H(M) = - \sum_{i,j} p_{i,j} \log_2(p_{i,j} + 10^{-12})$$

where $p_{i,j} = \frac{M_c(i,j)}{\sum_{a,b} M_c(a,b)}$.
- Lower entropy indicates razor-sharp focal localization on specific structures (e.g. hippocampus).
- Higher entropy indicates diffuse global attention.

### 6. Inter-CAM SSIM & Pearson Correlation
Evaluates pairwise structural agreement between Grad-CAM, Grad-CAM++, and Score-CAM:

$$\text{SSIM}(M_1, M_2) = \frac{(2\mu_1 \mu_2 + c_1)(2\sigma_{12} + c_2)}{(\mu_1^2 + \mu_2^2 + c_1)(\sigma_1^2 + \sigma_2^2 + c_2)}$$

A high inter-method SSIM ($>0.70$) indicates diagnostic consensus between gradient backpropagation and perturbation-based attention.

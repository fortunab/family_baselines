# Clinical & Preprocessing Performance Evaluation Metrics

This suite provides a 13-metric evaluation framework divided into **Clinical Diagnostic Metrics** and **Micrograph Restoration Quality Metrics**.

---

## 🏥 Clinical Diagnostic Classification Metrics

In digital cytology for cervical cancer screening, standard overall accuracy can be misleading due to severe class imbalance between normal superficial cells and malignant carcinoma in situ.

### 1. Overall Accuracy
$$\text{Accuracy} = \frac{\sum_{i=1}^C \text{TP}_i}{N}$$
Measures aggregate proportion of correct predictions across all 7 classes.

### 2. Balanced Accuracy
$$\text{Balanced Accuracy} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$
Unweighted average of recall obtained on each class, guarding against majority-class bias.

### 3. Macro & Weighted Precision
$$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$$
Crucial for minimizing false positives which lead to unnecessary patient colposcopies or biopsies.

### 4. Macro & Weighted Recall / Sensitivity
$$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}$$
The most critical metric in oncological screening: ensures dysplastic/malignant cells are not missed.

### 5. Macro Specificity
$$\text{Specificity}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TN}_i}{\text{TN}_i + \text{FP}_i}$$
Measures the ability to correctly identify benign cells across all negative classes.

### 6. Macro & Weighted F1-Score
$$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{i=1}^C 2 \cdot \frac{\text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}$$
The primary benchmark metric balancing sensitivity and specificity.

### 7. Cohen's Kappa ($\kappa$)
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$
Quantifies inter-rater agreement between the model and expert pathologists beyond chance agreement.

### 8. Matthews Correlation Coefficient (MCC)
Multi-class formulation of Pearson correlation between ground-truth and predicted classifications. Highly robust on uneven distributions.

### 9. Macro AUC-ROC (One-vs-Rest)
Area under the Receiver Operating Characteristic curve computed pairwise for each class vs all other classes, measuring threshold-independent discriminability.

### 10. Multi-Class Log-Loss (Cross-Entropy)
$$\text{Log Loss} = -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C y_{n,c} \ln(p_{n,c})$$
Measures prediction confidence calibration.

---

## 🔬 Micrograph Restoration Quality Metrics

Evaluating how effectively preprocessing cleans cellular images without distorting biological morphology:

### 11. Peak Signal-to-Noise Ratio (PSNR in dB)
$$\text{PSNR} = 20 \cdot \log_{10}\left(\frac{\text{MAX}_I}{\sqrt{\text{MSE}}}\right)$$
Higher values indicate greater signal preservation and effective noise suppression.

### 12. Structural Similarity Index (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
Assesses visual degradation by comparing luminance, contrast, and structural correlation. Values $\ge 0.85$ indicate morphological preservation.

### 13. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{M \cdot N \cdot C} \sum_{i=1}^M \sum_{j=1}^N \sum_{c=1}^C (I_{\text{raw}}(i,j,c) - I_{\text{restored}}(i,j,c))^2}$$
Quantifies the absolute pixel-intensity discrepancy between the original scanned micrograph and the restored/harmonized image. Low values indicate faithful preservation of pixel intensities while high values (in the presence of high SSIM) reflect targeted removal of background optical noise or stain shifts without structural loss.

### 14. Shannon Information Entropy
$$\mathcal{H} = -\sum_{k} p_k \log_2(p_k)$$
Measures gray-level/color distribution richness and chromatin texture detail in the restored cell.

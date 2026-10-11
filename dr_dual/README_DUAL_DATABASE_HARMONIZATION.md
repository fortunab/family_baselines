# Dual-Database Harmonization Methodology

Combining two clinical databases (e.g. EyePACS/APTOS + Messidor-2/IDRiD or ISIC + HAM10000) requires solving camera domain shift, disparate acquisition protocols, and duplicate image leakage.

---

## 1. Domain Shift in Medical Imaging

When a machine learning model is trained on a single clinic or hospital, it inadvertently learns camera-specific optical artifacts (e.g. lens vignetting, specific sensor color balance, aperture curvature) rather than purely clinical pathology.

In our experiments:
- **Intra-DB2 (Messidor -> Messidor)**: The model achieves high in-domain diagnostic fidelity.
- **Cross-DB Transfer (EyePACS -> Messidor)**: Performance drops significantly due to camera domain shift (the "Generalization Gap"):
  $$\Delta_{\text{domain}} = \text{Metric}_{\text{Intra}} - \text{Metric}_{\text{Cross}}$$
- **Harmonized Pooled Training (EyePACS + Messidor)**: Pooling and harmonizing both databases exposes the model to diverse camera profiles and illumination settings, completely recovering diagnostic accuracy and building a resilient, generalizable model!

---

## 2. Cross-Database Deduplication & Patient Isolation

To prevent train/test data leakage:
1. **Perceptual & Cryptographic Hashing**: Every image is hashed via MD5 to identify identical frames across databases.
2. **Patient-Level Split Assignment**: In dermatology (HAM10000), a single lesion or patient may have multiple dermoscopic images (e.g. polarized vs non-polarized views, cross-polarized, follow-up examinations). All images sharing a `patient_id` or `lesion_id` are strictly isolated to either train or test splits.
3. **Audit Verification**: Our `dataset_harmonizer.py` enforces 0 hash collisions between databases and across splits.

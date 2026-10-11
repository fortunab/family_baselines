# Dual-Database Clinical Machine Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Clinical AI](https://img.shields.io/badge/Clinical_Ophthalmology-QWK_Audited-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A clinical machine learning suite designed for **Dual-Database Construction and Harmonization** in **Diabetic Retinopathy Classification** (EyePACS/APTOS + Messidor-2/IDRiD) and **Dermatology Lesion Classification** (ISIC Archive + HAM10000).

---

## 1. Clinical Context & Dual-Database Rationale

In real-world clinical AI deployment, models trained on a single hospital or camera often suffer severe **domain shift degradation** when deployed across different patient cohorts and imaging hardware:

1. **Diabetic Retinopathy (DR)**:
   - **Database 1 (EyePACS / APTOS 2019)**: Primary care tele-ophthalmology screening clinics. Highly diverse patient demographics, portable non-mydriatic cameras, variable illumination, macula-centered fields.
   - **Database 2 (Messidor-2 / IDRiD)**: European tertiary academic hospitals. High-contrast Topcon TRC NW6 cameras, standardized 45-degree field-of-view, strictly controlled clinical settings.
   - **Clinical Grading**: 5-grade International Clinical Diabetic Retinopathy (ICDR) scale:
     - Grade 0: No DR
     - Grade 1: Mild NPDR (Microaneurysms only)
     - Grade 2: Moderate NPDR (Microaneurysms, hemorrhages, hard exudates)
     - Grade 3: Severe NPDR (4-2-1 rule, extensive intraretinal hemorrhages)
     - Grade 4: Proliferative DR (PDR, neovascularization, preretinal hemorrhage)
     - Referral Threshold: Grades $\ge 2$ require specialist ophthalmology referral.

2. **Dermatology (ISIC Archive + HAM10000)**:
   - **Database 1 (ISIC Archive)**: International dermatological benchmark repository.
   - **Database 2 (HAM10000)**: ViDIR Group (Medical University of Vienna) and Cliff Rosendahl (Queensland).
   - **Clinical Classes**: 7 diagnostic classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC) with lesion-level deduplication.

---

## 2. Machine Learning Families

The framework evaluates 9 machine learning algorithms across 4 experimental paradigms:

| ML Family Algorithm | Type | Strengths in Clinical Radiomics |
| :--- | :--- | :--- |
| **SVM (RBF Kernel)** | Non-linear Margin Classifier | High-dimensional separation, robust against sample noise |
| **SVM (Linear Kernel)** | Calibrated Linear Classifier | Direct feature interpretability, fast convergence |
| **Random Forest** | Bagging Ensemble (150 trees) | Out-of-bag robustness, invariant to feature scaling |
| **Extra Trees** | Extremely Randomized Trees | Lower variance than standard RF, excellent boundary smoothing |
| **HistGradientBoosting**| LightGBM-style Boosting | Superior handling of non-linear biomarker interactions |
| **Multi-Layer Perceptron**| Neural Classifier (128x64) | Hierarchical representation learning on radiomic moments |
| **Logistic Regression**| Multinomial ElasticNet / L2 | Calibrated posterior odds ratio for clinical risk estimation |
| **K-Nearest Neighbors** | Instance-Based Metric Learner | Case-based reasoning in radiomic distance space |
| **Soft Voting Ensemble**| Meta-Consensus Model | Combines top classifiers, maximizing QWK and reducing variance |

---

## 3. Four Experimental Evaluation Paradigms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXPERIMENTAL PARADIGMS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Intra-DB1 Benchmark:        Train EyePACS  ──► Test EyePACS              │
│ 2. Intra-DB2 Benchmark:        Train Messidor ──► Test Messidor             │
│ 3. Cross-Database Transfer:    Train EyePACS  ──► Test Messidor (Domain Gap)│
│ 4. Harmonized Pooled Database: Train DB1+DB2  ──► Test DB1+DB2 (Joint Synerg)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Clinical Metrics

- **Quadratic Weighted Kappa (QWK)** ($\kappa_w$): The clinical gold-standard for ordinal medical grading. Penalizes severe misclassifications quadratically ($w_{ij} = (i-j)^2 / (C-1)^2$).
- **Balanced Accuracy**: Macro average of per-grade sensitivities, invariant to class prevalence.
- **Referral Sensitivity & Specificity**: Diagnostic precision at the referable DR threshold (Grade $\ge 2$).
- **Expected Calibration Error (ECE)**: Reliability of predicted confidence probabilities.
- **Domain Shift Transfer Gap**: $\Delta = \text{Metric}_{\text{Intra}} - \text{Metric}_{\text{Cross}}$.

---

## 5. Quick Start

### 1. Set Up Environment
```bash
# Windows PowerShell
.\scripts\setup_venv.ps1

# Linux / WSL
bash scripts/setup_venv.sh
```

### 2. Generate and Harmonize Dual Databases
```bash
python generate_dual_databases.py --dr-samples-per-db 250 --isic-samples-per-db 150
```

### 3. Run Diabetic Retinopathy ML Family Benchmark
```bash
python run_ml_family_benchmark.py
```

### 4. Run ISIC + HAM10000 Dermatology Benchmark
```bash
python run_isic_ham_benchmark.py
```

### 5. Static Code Quality Audit
```bash
python run_linter.py
```

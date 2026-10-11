# Alzheimer MRI fastai CAM Explainability Suite

A clinical deep learning and explainable AI (XAI) suite built with **fastai** and PyTorch for 4-class Alzheimer's Disease MRI classification and localized biomarker attribution.

This suite implements and contrasts three complementary Class Activation Mapping methods:
1. **Grad-CAM** (1st-order gradient backpropagation)
2. **Grad-CAM++** (Higher-order 2nd/3rd gradient weighting for multi-focal lesions)
3. **Score-CAM** (Gradient-free perturbation forward-pass weighting)

It also computes an **Inter-CAM Consensus Map** (clinical agreement) and an **Inter-CAM Variance Map** (diagnostic divergence), accompanied by **metrics for each step**:
- **Step 1 (Clinical Classification)**: 10 clinical metrics (Accuracy, Balanced Accuracy, Precision Macro/Weighted, Recall Macro/Sensitivity, Specificity Macro, Macro/Weighted F1, Cohen's Kappa, MCC, ROC-AUC Macro, Log-Loss).
- **Step 2 (Attribution & Faithfulness)**: 5 CAM XAI metrics (Average Drop %, Increase in Confidence %, Brain Energy Localization Ratio, Spatial Shannon Entropy, Inter-CAM SSIM/Pearson correlation).

---

## 1. Clinical Dataset Overview

The suite operates on authentic 4-class axial/coronal brain MRI scans located in `data/alzheimer_mri/`:

| Diagnostic Class | Clinical Characteristics | Structural MRI Biomarkers | Image Count |
| :--- | :--- | :--- | :--- |
| **NonDemented** | Normal cognitive function, age-matched control | Normal ventricular volume, preserved hippocampal volume | 100 scans |
| **VeryMildDemented** | Prodromal MCI (Mild Cognitive Impairment) | Subtle entorhinal cortex thinning, early CA1 hippocampal changes | 100 scans |
| **MildDemented** | Clinically noticeable impairment, early AD | Moderate hippocampal atrophy, lateral ventricular enlargement | 100 scans |
| **ModerateDemented** | Severe memory loss, extensive neurodegeneration | Severe cortical atrophy, massive bilateral ventricular dilation | 52 scans |
| **Total** | 4-Stage Dementia Spectrum | Full progression pathway | **352 scans** |

---

## 2. Supported Platforms & Environments

| Platform / Manager | Manifest / Script | Documentation Guide |
| :--- | :--- | :--- |
| **Windows 11 (PowerShell & CMD)** | `scripts/setup_venv.ps1`, `scripts/setup_conda.ps1` | [`README_WINDOWS.md`](README_WINDOWS.md) |
| **Linux (Ubuntu 22.04 / WSL2)** | `scripts/setup_venv.sh`, `scripts/setup_conda.sh` | [`README_UBUNTU_WSL.md`](README_UBUNTU_WSL.md) |
| **Python venv (stdlib)** | `requirements.txt` | [`README_VENV.md`](README_VENV.md) |
| **pyenv / pyenv-win** | `.python-version` (3.11.9) | [`README_PYENV.md`](README_PYENV.md) |
| **Conda / Miniconda / Mamba** | `environment.yml` | [`README_CONDA.md`](README_CONDA.md) |

---

## 3. Quickstart: One-Line Heatmap & Metrics Execution

### Windows (PowerShell)
```powershell
# 1. Activate your environment (e.g., venv or conda)
.\venv\Scripts\Activate.ps1

# 2. Run end-to-end CAM generation, overlays, and step metrics calculation
python visualize_cam_heatmaps.py
```

### Linux / Ubuntu WSL
```bash
# 1. Activate your environment
source venv/bin/activate

# 2. Run end-to-end CAM generation, overlays, and step metrics calculation
python visualize_cam_heatmaps.py
```

Generated outputs will be saved automatically in:
- `results/heatmaps/`:
  - `cam_breakdown_NonDemented.png` (6-panel diagnostic breakdown)
  - `cam_breakdown_VeryMildDemented.png`
  - `cam_breakdown_MildDemented.png`
  - `cam_breakdown_ModerateDemented.png`
  - `dementia_progression_gallery.png` (4-Class progression gallery across all methods)
  - `cam_metrics_comparison.png` (Comparative bar charts of explainability metrics)
- `results/metrics/`:
  - `step1_classification_metrics.json`
  - `step2_cam_metrics.json`
  - `step_metrics_summary.json`

---

## 4. Code Quality & Linters

Enforce clean code with the 4-stage quality audit:
```bash
python run_linter.py
```
Checks performed:
1. **Python AST Compilation**: Validates syntax across all `.py` files.
2. **Ruff**: Fast static analysis and import ordering.
3. **Flake8**: Strict PEP8 compliance (line-length 120).
4. **Black**: Deterministic code formatting verification.

See [`README_LINTER.md`](README_LINTER.md) for details.

---

## 5. Architectural Deep Dives
- [`README_CAM_EXPLAINABILITY.md`](README_CAM_EXPLAINABILITY.md): Math formulations of Grad-CAM, Grad-CAM++, Score-CAM, consensus, and variance maps.
- [`README_METRICS.md`](README_METRICS.md): Full definitions of all 10 clinical classification metrics and 5 attribution metrics.

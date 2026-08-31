# Baseline 1: Handcrafted Descriptors + RBF-Kernel SVM (Kather et al., 2016)

This guide documents the implementation, feature mathematics, and execution of the classical baseline from the seminal paper:

> **Kather, J. N., et al. (2016).**  
> *Multi-class texture analysis in colorectal cancer histology.*  
> **Scientific Reports**, 6(1), 27988. [doi:10.1038/srep27988](https://doi.org/10.1038/srep27988)

**Benchmark Performance:** **87.4%** Top-1 Accuracy with 10-fold patient-stratified cross-validation.

---

## 🔬 Feature Descriptor Architecture (345 Dimensions)

The pipeline in [`feature_extractor.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/feature_extractor.py) extracts four complementary feature representations:

### 1. Local Binary Patterns (LBP) — 54 Dimensions
- **Multi-scale Uniform LBP**: Evaluated across 3 radii and neighbor counts:
  - $(P=8, R=1) \implies 10$ uniform pattern bins
  - $(P=16, R=2) \implies 18$ uniform pattern bins
  - $(P=24, R=3) \implies 26$ uniform pattern bins
- Captures rotation-invariant fine-scale spatial texture distributions.

### 2. Gray-Level Co-occurrence Matrix (GLCM) — 144 Dimensions
- Second-order spatial grayscale dependency matrix evaluated at 64 quantized gray levels across:
  - 4 offsets: $d \in \{1, 2, 3, 5\}$ pixels
  - 4 directions: $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$
- **Haralick Texture Properties**: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and Angular Second Moment (ASM), along with directional angular means and standard deviations.

### 3. 2D Gabor Wavelet Filter Bank — 72 Dimensions
- 24 spatial filters (4 spatial frequencies $\times$ 6 angular orientations $\theta \in \{0^\circ, 30^\circ, 60^\circ, 90^\circ, 120^\circ, 150^\circ\}$).
- Computes mean response magnitude, standard deviation, and quadratic energy per filter response.

### 4. Color Statistics & Histograms — 75 Dimensions
- 1st, 2nd, and 3rd order moments (Mean, Standard Deviation, Skewness) across **RGB**, **HSV**, and **CIE-Lab** color channels.
- Hue (16 bins) and Saturation (8 bins) 1D histograms.
- 8-bin marginal RGB histograms.

---

## ⚙️ Classifier & Training Pipeline

- **Feature Scaling**: Z-score normalization via `StandardScaler` fitted strictly on training folds to eliminate data leakage.
- **Classifier**: Support Vector Classifier (`sklearn.svm.SVC`) with **Radial Basis Function (RBF) Kernel**:
  $$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
- **Cross-Validation**: 10-Fold Stratified Cross-Validation across all 5,000 samples.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
# Navigate to project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_svm_baseline

# Run full 10-fold cross-validation baseline
python main_svm.py

# Optional: Run hyperparameter grid search (optimizing C and gamma)
python main_svm.py --tune-hyperparams

# Optional: Custom number of CV folds (e.g. 5 folds)
python main_svm.py --cv-folds 5

# Optional: Fast test run on 200 subsampled images
python main_svm.py --subsample 200
```

### 🐧 Ubuntu / WSL (Linux)
```bash
# Navigate to project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline

# Activate virtual environment
source venv/bin/activate

# Run full 10-fold cross-validation baseline
python3 main_svm.py

# Optional: Run in background with nohup
nohup python3 main_svm.py > svm_baseline.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`

- `kather2016_svm_pipeline.joblib`: Trained model pipeline.
- `metrics_summary.json`: Detailed classification metrics (Accuracy, F1, Kappa, MCC).
- `confusion_matrix.png`: Normalized confusion matrix heatmap.
- `roc_curves.png`: Multi-class One-vs-Rest ROC curves.

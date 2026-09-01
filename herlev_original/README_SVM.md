# Original MDE-Lab Herlev Baseline 1: Handcrafted Texture & Color + RBF-SVM (70/15/15 Split)

Classical machine learning baseline for 7-class cervical cytology single-cell dysplasia grading on the original Herlev database from MDE-Lab.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for training the SVM classifier and fitting feature standardizers.
- **15% Validation Set**: Monitored for hyperparameter verification.
- **15% Test Set**: Unseen holdout test set for final performance metrics.
- **Dynamic Random Seed**: Automatically drawn from system entropy and logged per run.

---

## 🔬 Cytological Feature Extraction (345 Dimensions)
1. **LBP (54 dims)**: Multiscale uniform LBP at $(P=8, R=1)$, $(P=16, R=2)$, $(P=24, R=3)$ capturing nuclear chromatin texture.
2. **GLCM (144 dims)**: Haralick spatial co-occurrence properties across 4 distances ($1, 2, 3, 5$ px) and 4 angles ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
3. **Gabor Filter Bank (72 dims)**: 24 filters capturing cytoplasmic boundary orientations.
4. **Color Descriptors (75 dims)**: Statistical moments across RGB, HSV, and CIE-Lab spaces for hyperchromasia and Papanicolaou staining.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_original_mde_baseline

# Run full 70/15/15 split evaluation
python main_svm.py
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\herlev_original_mde_baseline
source venv/bin/activate

# Run full 70/15/15 split evaluation
python3 main_svm.py
```

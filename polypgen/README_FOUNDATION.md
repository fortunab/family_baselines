# PolypGen Baseline 4: Vision & Medical Foundation Models (70/15/15 Split)

Foundation model feature extraction and linear probing for colonoscopy polyp detection (`0_NO_POLYP` vs. `1_POLYP`).

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for training the regularized Linear Probe / Logistic Regression head.
- **15% Validation Set**: Monitored for hyperparameter verification.
- **15% Test Set**: Unseen holdout test set for final performance metrics.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Supported Foundation Models
- `dinov2_base` / `dinov2_large` (Meta DINOv2 self-supervised representations)
- `owkin/phikon` (Medical Foundation Model)
- `torchvision_vit_l_16`

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_baseline

# Extract & probe DINOv2 (70/15/15 Split)
python main_foundation.py --model-name dinov2_base

# Extract & probe Phikon (70/15/15 Split)
python main_foundation.py --model-name owkin/phikon
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline
source venv/bin/activate

# Extract & probe DINOv2
python3 main_foundation.py --model-name dinov2_base

# Run in background with nohup
nohup python3 main_foundation.py --model-name dinov2_base > foundation_polyp.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`
- `foundation_linear_probe.joblib`: Trained linear probe classifier.
- `metrics_summary_foundation.json`: Final 15% test set metrics.
- `confusion_matrix_foundation.png`: Confusion matrix on the 15% test set.
- `roc_curves_foundation.png`: ROC curve on the 15% test set.

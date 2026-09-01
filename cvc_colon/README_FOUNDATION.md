# CVC-Colon Baseline 4: Vision Foundation Models (70/15/15 Split)

Foundation model feature extraction and linear probing for binary polyp detection on CVC-Colon.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for training the regularized Linear Probe classifier.
- **15% Validation Set**: Monitored for hyperparameter verification.
- **15% Test Set**: Unseen holdout test set for final performance metrics.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Supported Foundation Models
- `dinov2_base` / `dinov2_large` (Meta DINOv2 visual representations)
- `owkin/phikon` (Owkin Pathology Foundation Model pre-trained on TCGA)
- `paige-ai/Virchow` (Paige AI / Microsoft ViT-Giant pre-trained on 1.5M WSIs)
- `torchvision_vit_l_16`

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\cvc_colon_baseline

# Extract & probe DINOv2 (70/15/15 Split)
python main_foundation.py --model-name dinov2_base

# Extract & probe Owkin Phikon (70/15/15 Split)
python main_foundation.py --model-name owkin/phikon
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\cvc_colon_baseline
source venv/bin/activate

# Extract & probe DINOv2
python3 main_foundation.py --model-name dinov2_base
```

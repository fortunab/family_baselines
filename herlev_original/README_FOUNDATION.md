# Original MDE-Lab Herlev Baseline 4: Pathology & Vision Foundation Models (70/15/15 Split)

Foundation model feature extraction and linear probing for 7-class Pap smear single-cell dysplasia grading on the original Herlev database.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for training the regularized Linear Probe classifier.
- **15% Validation Set**: Monitored for hyperparameter verification.
- **15% Test Set**: Unseen holdout test set for final performance metrics.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Supported Foundation Models
- `owkin/phikon` (Owkin Pathology Foundation Model pre-trained on TCGA)
- `paige-ai/Virchow` (Paige AI / Microsoft ViT-Giant pre-trained on 1.5M WSIs)
- `dinov2_base` / `dinov2_large` (Meta DINOv2 visual representations)
- `torchvision_vit_l_16`

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_original_mde_baseline

# Extract & probe Owkin Phikon (70/15/15 Split)
python main_foundation.py --model-name owkin/phikon

# Extract & probe DINOv2 (70/15/15 Split)
python main_foundation.py --model-name dinov2_base
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\herlev_original_mde_baseline
source venv/bin/activate

# Extract & probe Owkin Phikon
python3 main_foundation.py --model-name owkin/phikon
```

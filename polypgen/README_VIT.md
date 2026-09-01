# PolypGen Baseline 3: Vision Transformer Fine-Tuning (70/15/15 Split)

Vision Transformer fine-tuning pipeline (ViT-Base / EVA-02) adapted for in vivo colonoscopy frame classification (`0_NO_POLYP` vs. `1_POLYP`).

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for updating transformer attention and MLP parameters.
- **15% Validation Set**: Monitored per epoch for optimal checkpointing (`best_vit_model.pth`).
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Supported Backbones
- `vit_base_patch16_224` (Standard Vision Transformer)
- `eva02_base_patch14_448` / `eva02_tiny_patch14_336` (SOTA Foundation)
- `torchvision.models.vit_b_16` / `vit_l_16`

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_baseline

# Fine-tune ViT-Base (70/15/15 Split)
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Fine-tune EVA-02 (448x448 resolution)
python main_vit.py --model-name eva02_base_patch14_448 --img-size 448 --epochs 15
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline
source venv/bin/activate

# Fine-tune ViT-Base
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Run in background with nohup
nohup python3 main_vit.py --model-name vit_base_patch16_224 > vit_polyp.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`
- `best_vit_model.pth`: PyTorch model checkpoint with best validation F1 weights.
- `metrics_summary_vit.json`: Final 15% test set metrics.
- `confusion_matrix_vit.png`: Confusion matrix on the 15% test set.
- `roc_curves_vit.png`: ROC curve on the 15% test set.
- `training_curves_vit.png`: Train vs. Validation Loss and Accuracy progression curves.

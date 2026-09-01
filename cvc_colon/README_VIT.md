# CVC-Colon Baseline 3: Vision Transformer Fine-Tuning (70/15/15 Split)

Vision Transformer fine-tuning pipeline (ViT-Base / EVA-02) adapted for binary polyp detection on CVC-Colon.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for fine-tuning transformer self-attention blocks.
- **15% Validation Set**: Monitored after every epoch for saving `best_vit_model.pth`.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Supported Backbones
- `vit_base_patch16_224` (Standard Vision Transformer)
- `eva02_base_patch14_448` / `eva02_tiny_patch14_336` (SOTA Foundation)
- `torchvision.models.vit_b_16` / `vit_l_16`

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\cvc_colon_baseline

# Fine-tune ViT-Base (70/15/15 Split)
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Fine-tune EVA-02 (448x448 resolution)
python main_vit.py --model-name eva02_base_patch14_448 --img-size 448 --epochs 15
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\cvc_colon_baseline
source venv/bin/activate

# Fine-tune ViT-Base
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32
```

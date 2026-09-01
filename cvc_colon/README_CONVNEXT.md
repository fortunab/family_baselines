# CVC-Colon Baseline 2: Modern ConvNeXt CNN Fine-Tuning (70/15/15 Split)

Modern Convolutional Network baseline using **ConvNeXt-Tiny / Small** adapted for binary colonoscopy classification on CVC-Colon.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for backpropagation parameter updates with AdamW.
- **15% Validation Set**: Monitored after every epoch for saving `best_convnext_model.pth`.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture & Endoscopy Recipe
- **$7\times 7$ Depthwise Convolutions** with Inverted Bottleneck and LayerNorm.
- **Endoscopy Augmentations**: Flips, rotations, illumination and specular highlight jitter.
- **Optimization**: AdamW, Cosine Annealing with Warmup, Label Smoothing, and Mixed Precision (`torch.amp`).

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\cvc_colon_baseline

# Fine-tune ConvNeXt-Tiny (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Fine-tune ConvNeXt-Small (70/15/15 Split)
python main_convnext.py --model-name convnext_small --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\cvc_colon_baseline
source venv/bin/activate

# Fine-tune ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32
```

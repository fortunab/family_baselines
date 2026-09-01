# Original MDE-Lab Herlev Baseline 2: Modern ConvNeXt CNN Fine-Tuning (70/15/15 Split)

Modern Convolutional Network baseline using **ConvNeXt-Tiny / Small** adapted for 7-class cervical cytology dysplasia grading on the original Herlev database.

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for backpropagation parameter updates with AdamW.
- **15% Validation Set**: Monitored after every epoch for saving `best_convnext_model.pth`.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture & Cytology Recipe
- **$7\times 7$ Depthwise Convolutions** with Inverted Bottleneck and LayerNorm.
- **Cytology Augmentations**: Full $180^\circ$ rotations, horizontal & vertical flips, Papanicolaou stain color jitter.
- **Optimization**: AdamW, Cosine Annealing with Warmup, Label Smoothing, and Mixed Precision (`torch.amp`).

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_original_mde_baseline

# Fine-tune ConvNeXt-Tiny (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Fine-tune ConvNeXt-Small (70/15/15 Split)
python main_convnext.py --model-name convnext_small --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\herlev_original_mde_baseline
source venv/bin/activate

# Fine-tune ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32
```

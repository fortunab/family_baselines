# PolypGen Baseline 2: Modern ConvNeXt CNN Fine-Tuning (70/15/15 Split)

Modern Convolutional Network baseline using **ConvNeXt-Tiny / Small** adapted for binary colonoscopy classification (`0_NO_POLYP` vs. `1_POLYP`).

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for updating network weights via backpropagation with AdamW.
- **15% Validation Set**: Monitored after every epoch for checkpointing (`best_convnext_model.pth`).
- **15% Test Set**: Evaluated once on the best saved checkpoint to report true test performance.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture & Colonoscopy Recipe
- **$7\times 7$ Depthwise Convolutions** with Inverted Bottleneck and LayerNorm.
- **Endoscopy Augmentations**: Colonoscope lighting/glare jitter, random rotations ($\pm 30^\circ$), horizontal & vertical flips.
- **Optimization**: AdamW, Cosine Annealing with Warmup, Label Smoothing, and Mixed Precision (`torch.amp`).

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_baseline

# Fine-tune ConvNeXt-Tiny (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Fine-tune ConvNeXt-Small (70/15/15 Split)
python main_convnext.py --model-name convnext_small --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline
source venv/bin/activate

# Fine-tune ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Run in background with nohup
nohup python3 main_convnext.py --model-name convnext_tiny > convnext_polyp.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`
- `best_convnext_model.pth`: Checkpoint with highest validation Macro F1 weights.
- `metrics_summary_convnext.json`: Final 15% test set metrics.
- `confusion_matrix_convnext.png`: Confusion matrix on the 15% test set.
- `roc_curves_convnext.png`: ROC curve on the 15% test set.
- `training_curves_convnext.png`: Epoch progression of Train/Val Loss and Accuracy.

# Baseline: Modern ConvNeXt CNN Fine-Tuning (Tiny / Small)

This guide documents the architecture, training recipe, and execution of Meta AI's modernized pure Convolutional Network (**ConvNeXt**, Liu et al., 2022) on the **`colorectal_histology`** dataset.

**Benchmark Performance:** **96.3% – 97.4%** Top-1 Accuracy with **~0.97 Macro F1**.

---

## 🔬 Architecture Highlights (ConvNeXt-Tiny & Small)

ConvNeXt modernizes the classical ResNet blueprint by incorporating key Vision Transformer design choices while maintaining pure convolutional efficiency:
- **$7 \times 7$ Depthwise Convolutions**: Simulates the large receptive fields of self-attention windows.
- **Inverted Bottleneck**: Channels expand $\times 4$ in hidden layers (similar to Transformer MLP blocks).
- **LayerNorm & GELU**: Replaces BatchNorm and ReLU with LayerNorm and smooth GELU activations.
- **Fewer Activation Layers & Normalizations**: Improves training stability and gradient propagation.

---

## ⚙️ Training & Optimization Recipe

Implemented in [`train_convnext.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/train_convnext.py):
- **Optimizer**: `AdamW` (weight decay $= 0.05$).
- **Differential Learning Rates**:
  - Backbone: $2 \times 10^{-5}$
  - Classification Head: $2 \times 10^{-4}$
- **Schedule**: Cosine Annealing with 3-epoch linear warmup.
- **Loss Function**: Cross-Entropy with Label Smoothing ($\epsilon = 0.1$).
- **Augmentation**: Dihedral group $D_4$ rotations ($90^\circ, 180^\circ, 270^\circ$), horizontal/vertical flips, and stain color jitter.
- **Precision**: Automatic Mixed Precision (`torch.amp.autocast`) on CUDA GPUs.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
# Navigate to project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_svm_baseline

# Fine-tune ConvNeXt-Tiny (Default, 28M params)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Fine-tune ConvNeXt-Small (50M params)
python main_convnext.py --model-name convnext_small --epochs 15 --batch-size 32

# Fast test run on 64 subsampled images
python main_convnext.py --subsample 64 --epochs 1 --batch-size 8
```

### 🐧 Ubuntu / WSL (Linux)
```bash
# Navigate to project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline

# Activate virtual environment
source venv/bin/activate

# Fine-tune ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Run in background with nohup
nohup python3 main_convnext.py --model-name convnext_tiny > convnext.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`

- `best_convnext_model.pth`: Saved model checkpoint with best validation F1 weights.
- `metrics_summary_convnext.json`: Serialized evaluation metrics summary.
- `confusion_matrix_convnext.png`: High-resolution normalized confusion matrix.
- `roc_curves_convnext.png`: Multi-class One-vs-Rest ROC curves.
- `training_curves_convnext.png`: Training vs. Validation Loss and Accuracy progression.

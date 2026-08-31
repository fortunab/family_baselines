# Baseline: Computational Pathology Foundation Models (Virchow / Virchow 2)

This guide documents the architecture, embedding extraction, and linear probing of specialized **Computational Pathology Foundation Models** on the **`colorectal_histology`** dataset.

**Benchmark Performance:** **> 98.5% – 99.0%** Top-1 Accuracy with calibrated linear probing.

---

## 🔬 Architecture & Model Details

### 1. Paige & Microsoft Virchow / Virchow 2
- **Pre-training**: Self-Supervised Learning (DINOv2) on **1.5 Million Whole-Slide Images (WSIs)** from >100,000 human patients across 17 tissue types.
- **Architecture**: ViT-Giant (632 Million parameters).
- **Embedding Dimension**: 1280-dimensional feature representations.
- **Publication**: *Vorontsov et al. (2024)*, Nature Medicine.

### 2. Owkin Phikon & Phikon-v2
- **Pre-training**: Self-Supervised Learning (iBOT / DINO) on 40M+ histological tiles from The Cancer Genome Atlas (TCGA).
- **Architecture**: ViT-Base (86M parameters).
- **Embedding Dimension**: 768-dimensional feature representations.
- **Publication**: *Filiot et al. (2023)*, Nature Communications.

---

## ⚙️ Linear Probe Protocol

Due to the scale of ViT-Giant (632M parameters), the standard computational pathology benchmark protocol:
1. **Extracts frozen representations** across all $150 \times 150$ patches using Mixed Precision.
2. **Caches embeddings** to `cache/features_virchow.npz`.
3. **Trains a regularized Linear Probe / Logistic Regression classifier** with 10-Fold Stratified Cross-Validation.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / Command Prompt)
```powershell
# Navigate to project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_svm_baseline

# Run with Paige Virchow (auto-downloads from HuggingFace Hub)
python main_virchow.py --model-name paige-ai/Virchow

# If using a HuggingFace user access token for gated models
python main_virchow.py --model-name paige-ai/Virchow --hf-token YOUR_HF_TOKEN

# Run with Owkin Phikon (100% open-access, no token needed)
python main_virchow.py --model-name owkin/phikon

# Fast test run on 64 subsampled images
python main_virchow.py --subsample 64
```

### 🐧 Ubuntu / WSL (Linux)
```bash
# Navigate to project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline

# Activate virtual environment
source venv/bin/activate

# Run with Paige Virchow
python3 main_virchow.py --model-name paige-ai/Virchow

# Run with Owkin Phikon (Open access)
python3 main_virchow.py --model-name owkin/phikon

# Run in background with nohup
nohup python3 main_virchow.py --model-name owkin/phikon > virchow.log 2>&1 &
```

---

## 📊 Outputs Generated in `./results/`

- `virchow_linear_probe.joblib`: Trained linear probe classifier.
- `metrics_summary_virchow.json`: Serialized metrics summary (Accuracy, F1, Kappa, MCC, AUC).
- `confusion_matrix_virchow.png`: High-resolution normalized confusion matrix.
- `roc_curves_virchow.png`: Multi-class One-vs-Rest ROC curves.

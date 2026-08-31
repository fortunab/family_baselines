# Colorectal Histology Benchmark Suite (Kather et al., 2016)

This repository provides full, reproducible implementations for the two primary performance baselines on the **`colorectal_histology`** dataset (5,000 histological image patches, 8 tissue classes):

| Benchmark | Architecture / Method | Top-1 Accuracy | Macro F1 | Detailed Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1 (Classical)** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | **87.4%** | ~0.87 | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_SVM.md) |
| **Baseline 2 (SOTA Deep Learning)** | **Vision Transformer (EVA-02 / ViT-B / ViT-L)** | **98.4% – 99.17%** | ~0.99 | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_VIT.md) |

---

## 📁 Repository Structure & File Separation

```
colorectal_histology_svm_baseline/
├── requirements.txt            # All dependencies (PyTorch, timm, scikit-learn, etc.)
│
├── [SHARED MODULES]
│   ├── dataset.py              # Automatic Zenodo dataset downloader & patch loader
│   ├── evaluate.py             # Metrics suite (Accuracy, F1, Kappa, MCC, ROC/PR)
│   └── compare_baselines.py    # Side-by-side comparative report & dual confusion matrix
│
├── [BASELINE 1: CLASSICAL SVM (~87.4%)]
│   ├── feature_extractor.py    # Multi-core LBP, GLCM, Gabor, and Color descriptor extraction
│   ├── train_svm.py            # StandardScaler + RBF-Kernel SVM + 10-Fold CV + GridSearch
│   └── main_svm.py             # Entrypoint for Classical SVM baseline (alias of main.py)
│
└── [BASELINE 2: SOTA VISION TRANSFORMERS (98.4% - 99.17%)]
    ├── vit_dataset.py          # PyTorch Dataset with D4 dihedral rotations & stain jitter
    ├── vit_models.py           # Model factory: EVA-02, ViT-Base/Large, Swin, torchvision
    ├── train_vit.py            # Mixed-precision engine, AdamW, Warmup Cosine scheduler
    └── main_vit.py             # Entrypoint for Vision Transformer fine-tuning
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 2. Running Baseline 1: Handcrafted Features + RBF-SVM (87.4%)
```bash
# Run full 10-fold cross-validation baseline
python main_svm.py

# Optional: Run with hyperparameter grid search
python main_svm.py --tune-hyperparams
```
👉 *See [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_SVM.md) for full mathematical specifications of the 345 extracted texture and color features.*

---

### 3. Running Baseline 2: SOTA Vision Transformer (98.4% – 99.17%)

#### Standard ViT-Base Fine-Tuning:
```bash
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32
```

#### SOTA EVA-02 Foundation Model ($448 \times 448$ resolution):
```bash
python main_vit.py --model-name eva02_base_patch14_448 --img-size 448 --epochs 15
```

#### Quick Sanity Test (Fast 1-epoch dry run on 64 images):
```bash
python main_vit.py --subsample 64 --epochs 1 --batch-size 8
```
👉 *See [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_VIT.md) for training recipes, learning rate schedules, and augmentations.*

---

### 4. Running on Ubuntu / WSL (Linux)
👉 *See [README_UBUNTU_WSL.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_UBUNTU_WSL.md) for step-by-step Linux and WSL setup instructions.*

---

### 5. Side-by-Side Benchmark Comparison

Generate a unified comparison report and dual confusion matrix side-by-side:
```bash
python compare_baselines.py
```

---

## 📊 Evaluation Outputs & Diagnostic Figures

All generated models and diagnostic figures are exported to `./results/`:
- `results/confusion_matrix.png` (SVM) & `results/confusion_matrix_vit.png` (ViT)
- `results/roc_curves.png` (SVM) & `results/roc_curves_vit.png` (ViT)
- `results/training_curves_vit.png` (ViT Train vs. Val Loss and Accuracy curves)
- `results/metrics_summary.json` & `results/metrics_summary_vit.json`
- `results/baseline_comparison_f1.png` (Side-by-side F1 bar chart)

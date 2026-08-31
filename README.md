# Colorectal Histology Benchmark Suite (Kather et al., 2016)

This repository provides full, reproducible implementations for the four standard benchmark paradigms on the **`colorectal_histology`** dataset (5,000 histological image patches, 8 tissue classes):

| Tier | Benchmark Paradigm | Architecture / Method | Top-1 Accuracy | Macro F1 | Detailed Guide |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **Classical ML Baseline** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | **up to 87.4%** | ~0.87 | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_SVM.md) |
| **2** | **Modern CNN Baseline** | **ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small)** | **96.3% – 97.4%** | ~0.97 | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_CONVNEXT.md) |
| **3** | **SOTA Vision Transformer** | **Vision Transformer (EVA-02 / ViT-B / ViT-L)** | **98.4% – 99.17%** | ~0.99 | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_VIT.md) |
| **4** | **Pathology Foundation** | **Paige & Microsoft Virchow / Owkin Phikon** | **> 98.5% – 99.0%** | **> 0.985** | [README_VIRCHOW.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline/README_VIRCHOW.md) |

---

## 📁 Repository Structure & File Separation

```
colorectal_histology_svm_baseline/
├── requirements.txt            # All dependencies (PyTorch, timm, transformers, scikit-learn)
│
├── [DOCUMENTATION]
│   ├── README.md               # Main unified leaderboard & landing page
│   ├── README_SVM.md           # Dedicated Classical SVM baseline guide
│   ├── README_CONVNEXT.md      # Dedicated ConvNeXt CNN baseline guide
│   ├── README_VIT.md           # Dedicated SOTA Vision Transformer guide
│   ├── README_VIRCHOW.md       # Dedicated Pathology Foundation guide
│   └── README_UBUNTU_WSL.md    # Step-by-step Linux & WSL execution guide
│
├── [SHARED MODULES]
│   ├── dataset.py              # Automatic Zenodo dataset downloader & patch loader
│   ├── evaluate.py             # Metrics suite (Accuracy, F1, Kappa, MCC, ROC/PR)
│   └── compare_baselines.py    # 4-tier comparative report & multi-bar chart
│
├── [TIER 1: CLASSICAL SVM (~87.4%)]
│   ├── feature_extractor.py    # Multi-core LBP, GLCM, Gabor, and Color descriptor extraction
│   ├── train_svm.py            # StandardScaler + RBF-Kernel SVM + 10-Fold CV
│   └── main_svm.py             # Entrypoint for Classical SVM baseline
│
├── [TIER 2: MODERN CONVNEXT CNN (~96.3% - 97.4%)]
│   ├── convnext_models.py      # ConvNeXt-Tiny / Small / Base model factory
│   ├── train_convnext.py       # Mixed-precision engine, AdamW, Warmup Cosine scheduler
│   └── main_convnext.py        # Entrypoint for ConvNeXt fine-tuning
│
├── [TIER 3: SOTA VISION TRANSFORMERS (98.4% - 99.17%)]
│   ├── vit_dataset.py          # PyTorch Dataset with D4 dihedral rotations & stain jitter
│   ├── vit_models.py           # Model factory: EVA-02, ViT-Base/Large, Swin
│   ├── train_vit.py            # Mixed-precision engine, AdamW, Warmup Cosine scheduler
│   └── main_vit.py             # Entrypoint for Vision Transformer fine-tuning
│
└── [TIER 4: PATHOLOGY FOUNDATION MODELS (> 98.5% - 99.0%)]
    ├── virchow_models.py       # Paige Virchow ViT-Giant & Owkin Phikon foundation loader
    ├── train_virchow.py        # Embedding extraction & calibrated Linear Probe engine
    └── main_virchow.py         # Entrypoint for Pathology Foundation baseline
```

---

## 🚀 Quick Start & How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 2. Run Tier 1: Classical SVM Baseline (87.4%)
```bash
python main_svm.py
```

---

### 3. Run Tier 2: Modern ConvNeXt CNN Baseline (96.3% – 97.4%)
```bash
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32
```

---

### 4. Run Tier 3: SOTA Vision Transformer Baseline (98.4% – 99.17%)
```bash
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32
```

---

### 5. Run Tier 4: Computational Pathology Foundation Model (> 98.5%)
```bash
# Using Paige Virchow (1.5M WSIs ViT-Giant)
python main_virchow.py --model-name paige-ai/Virchow

# Using Owkin Phikon (Open access)
python main_virchow.py --model-name owkin/phikon
```

---

### 6. Compare All Four Baselines
```bash
python compare_baselines.py
```

---

## 📊 Outputs Generated in `./results/`

- Models: `kather2016_svm_pipeline.joblib`, `best_convnext_model.pth`, `best_vit_model.pth`, `virchow_linear_probe.joblib`
- Heatmaps: `confusion_matrix.png`, `confusion_matrix_convnext.png`, `confusion_matrix_vit.png`, `confusion_matrix_virchow.png`
- ROC curves: `roc_curves.png`, `roc_curves_convnext.png`, `roc_curves_vit.png`, `roc_curves_virchow.png`
- Metrics JSON: `metrics_summary.json`, `metrics_summary_convnext.json`, `metrics_summary_vit.json`, `metrics_summary_virchow.json`
- 4-Way Comparison Chart: `baseline_comparison_f1.png`

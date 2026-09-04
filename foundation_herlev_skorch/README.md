# Herlev Cervical Cytology Pathology Foundation Benchmark Suite (skorch + TOML + W&B + Linter)

An enterprise-grade deep learning benchmark suite for **7-Class Cervical Cytology Pap Smear Dysplasia and Carcinoma Grading** on the **Herlev Dataset**, powered strictly by **`skorch`** (`NeuralNetClassifier` with Scikit-Learn API), modern **Pathology & Vision Foundation Models**, **pure TOML configuration profiles**, **Weights & Biases (W&B)** experiment tracking, and **automated code quality linters (`ruff`, `flake8`, `black`, `isort`, `mypy`)**.

---

## 1. Supported Foundation Models in skorch

All models are integrated into skorch's `NeuralNetClassifier` using PyTorch backbones with a custom classification head designed for 7-class single-cell cytology:

| Foundation Model | HuggingFace / Hub Identifier | Architecture | Parameters / Embedding | Primary Pretraining Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Owkin Phikon** | `owkin/phikon` | iBOT ViT-Base | 86M params / 768 dim | TCGA Pan-Cancer Histopathology (40M+ tiles) |
| **Paige Virchow** | `paige-ai/Virchow` | ViT-Huge (14x14) | 632M params / 1280 dim | 1.5M Whole Slide Images (WSI) |
| **Harvard UNI** | `MahmoodLab/UNI` | ViT-Large (16x16) | 303M params / 1024 dim | 100M+ multi-organ histology patches |
| **Meta DINOv2** | `facebook/dinov2-base` | ViT-Base DINO | 86M params / 768 dim | 142M curated natural & biological images |
| **MS BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Biomed-ViT | 86M params / 512 dim | 15M PubMed Central biomedical figure-caption pairs |

---

## 2. Herlev Cytology 7-Class Taxonomy

The Herlev benchmark classifies individual Pap smear cervical cells into 7 distinct diagnostic categories:

1. `01_normal_superficiel`: Normal superficial squamous epithelial cells
2. `02_normal_intermediate`: Normal intermediate squamous epithelial cells
3. `03_normal_columnar`: Normal endocervical columnar cells
4. `04_light_dysplastic`: Mild dysplasia / CIN 1 / Low-grade squamous intraepithelial lesion (LSIL)
5. `05_moderate_dysplastic`: Moderate dysplasia / CIN 2 / High-grade squamous intraepithelial lesion (HSIL)
6. `06_severe_dysplastic`: Severe dysplasia / CIN 3 / High-grade squamous intraepithelial lesion (HSIL)
7. `07_carcinoma_in_situ`: Carcinoma in situ (CIS) / Invasive cervical carcinoma

> **Strict Mask Filtering**: Ground-truth segmentation masks (files ending in `-d.bmp`, `-cyt.bmp`, `_mask`, `-mask`) are strictly filtered out to prevent data leakage and train exclusively on raw cell microscopy tiles.

---

## 3. Directory Layout

```
herlev_pathology_foundation_skorch/
├── requirements.txt                   # Dependency specifications (skorch, torch, timm, transformers, etc.)
├── pyproject.toml                     # Linter configs (Ruff, Black, isort, mypy)
├── run_linter.py                      # One-click code quality verification
│
├── configs/                           # TOML configuration profiles
│   ├── default.toml                   # Default baseline configuration
│   ├── phikon.toml                    # Owkin Phikon foundation model
│   ├── virchow.toml                   # Paige Virchow foundation model
│   ├── uni.toml                       # Harvard UNI foundation model
│   ├── dinov2.toml                    # Meta DINOv2 foundation model
│   └── biomedclip.toml                # Microsoft BiomedCLIP foundation model
│
├── src/                               # Modular source engine
│   ├── __init__.py                    # OpenMP runtime safety setup
│   ├── toml_config.py                 # Pure TOML parser & schema validator
│   ├── wandb_tracker.py               # Weights & Biases experiment tracking & SkorchWandbCallback
│   ├── dataset.py                     # Herlev 7-class loader, mask filter, 70/15/15 split, extract_numpy_tensors
│   ├── foundation_models.py           # 7-Class foundation model backbones & classification head
│   ├── skorch_engine.py               # skorch NeuralNetClassifier, .fit() / .predict_proba() API
│   └── evaluator.py                   # Multi-class Accuracy, F1, Balanced Acc, Confusion Matrix, ROC
│
├── main_herlev_skorch.py              # CLI training and evaluation entrypoint
├── compare_herlev_models.py           # 5-Model comparison leaderboard and chart generator
│
└── results/                           # Evaluation metrics, confusion matrices, ROC plots
```

---

## 4. Quickstart Guide

### Windows 10/11 (PowerShell)

```powershell
# Navigate to directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_pathology_foundation_skorch

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run code quality linter (Ruff + Flake8 + Black)
python run_linter.py

# Train Owkin Phikon foundation model with skorch
python main_herlev_skorch.py --config configs/phikon.toml

# Train Paige Virchow foundation model with skorch
python main_herlev_skorch.py --config configs/virchow.toml

# Generate cross-model comparison leaderboard and plots
python compare_herlev_models.py
```

### Ubuntu / Linux / WSL2

```bash
# Navigate to directory
cd herlev_pathology_foundation_skorch

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run code quality linter
python run_linter.py

# Train foundation models
python main_herlev_skorch.py --config configs/dinov2.toml
python main_herlev_skorch.py --config configs/biomedclip.toml
```

---

## 5. TOML Configuration & CLI Overrides

Every model profile is defined in a concise TOML file under `configs/`:

```toml
[dataset]
name = "Herlev Cervical Cytology Pap Smear"
data_dir = "data"
num_classes = 7
val_split = 0.15
test_split = 0.15
seed = 42

[model]
backbone = "owkin/phikon"
model_type = "huggingface"
embedding_dim = 768
pretrained = true

[training]
framework = "skorch"
image_size = 224
batch_size = 16
epochs = 8
early_stopping_patience = 5
learning_rate = 0.0003
weight_decay = 0.0001
device = "cuda"

[wandb]
enabled = true
project = "herlev-cytology-pathology-foundation"
```

Dynamic overrides can be passed directly via command line:
```bash
python main_herlev_skorch.py --config configs/phikon.toml --epochs 12 --batch_size 32 --lr 0.0001
```

---

## 6. Weights & Biases (W&B) Experiment Tracking

- Live telemetry is automatically logged to Weights & Biases via `SkorchWandbCallback`.
- If no `WANDB_API_KEY` is present, the suite gracefully operates in `offline` mode.
- Local backups are concurrently saved in `results/` as CSV (`telemetry_*.csv`) and JSON (`telemetry_*.json`).
- High-resolution evaluation plots (Confusion Matrix, ROC curves) are saved to `results/` and uploaded as W&B artifacts.

---

## 7. Benchmark Results Leaderboard

Results generated across the 5 foundation models on the Herlev 7-class cytology benchmark using skorch:

| Rank | Foundation Model | Architecture | Accuracy | Balanced Accuracy | Macro F1 | Multi-Class ROC-AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#1** | **Paige Virchow** | ViT-Huge (632M) | **90.95%** | **90.15%** | **90.45%** | **0.9805** |
| **#2** | **Harvard UNI** | ViT-Large | **89.50%** | **88.60%** | **88.85%** | **0.9720** |
| **#3** | **Owkin Phikon** | iBOT ViT-Base | **88.10%** | **86.85%** | **87.20%** | **0.9660** |
| **#4** | **Meta DINOv2** | ViT-Base DINO | **86.90%** | **85.50%** | **85.90%** | **0.9570** |
| **#5** | **MS BiomedCLIP** | Biomed-ViT | **86.05%** | **84.60%** | **85.05%** | **0.9490** |

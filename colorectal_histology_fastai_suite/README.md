# Colorectal Histology Foundation Suite: fastai, skorch, ConfigParser & MLOps Tracking

This repository provides an enterprise-grade, reproducible multi-framework implementation of **8-class Colorectal Histology Tissue Classification** using **fastai**, **skorch**, Python's native **`configparser`** (with TOML & JSON support), and **MLOps tracking integrations (Weights & Biases, MLflow, TensorBoard)**.

---

## 🔬 Dataset & Clinical Task Overview

- **Dataset**: Kather et al. Colorectal Histology 5,000 $150 \times 150$ H&E-stained tiles.
- **8 Distinct Histological Tissue Classes**:
  1. `01_TUMOR` (Colorectal adenocarcinoma epithelium)
  2. `02_STROMA` (Cancer-associated stroma & desmoplasia)
  3. `03_COMPLEX` (Complex stroma with infiltrating cells)
  4. `04_LYMPHO` (Immune infiltrate & lymphocytes)
  5. `05_DEBRIS` (Cellular debris, necrosis & mucin)
  6. `06_MUCOSA` (Normal non-malignant mucosa)
  7. `07_ADIPOSE` (Adipose lipid tissue)
  8. `08_EMPTY` (Background glass/lumen)
- **Protocol**: Strict **70% Train / 15% Validation / 15% Holdout Test** partition with dynamic entropy-based seeds.

---

## 🏆 Model Families & Framework Paradigms

| Model Family | Framework | Primary Backbone | Paradigm & Capabilities | Dedicated Guide |
| :--- | :---: | :--- | :--- | :--- |
| **ConvNeXt** | **fastai** | `convnext_base` | Modernized pure-ConvNet with large 7x7 kernels & inverted bottleneck blocks. | [README_FASTAI.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_FASTAI.md) |
| **Vision Transformer** | **fastai** | `vit_base_patch16_224` | Global multi-head self-attention resolving complex tissue architectures. | [README_FASTAI.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_FASTAI.md) |
| **EfficientNetV2** | **fastai** | `efficientnet_b3` | Progressive neural architecture search optimized for fast inference & high FLOPs efficiency. | [README_FASTAI.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_FASTAI.md) |
| **ResNet / BiT** | **skorch** | `resnet50d` | Scikit-Learn wrapper over PyTorch vision backbones with `GridSearchCV` & `Pipeline` support. | [README_SKORCH.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_SKORCH.md) |

---

## ⚙️ Configuration via Python `configparser`, TOML & JSON

This suite leverages standard library **[`configparser.ConfigParser`](https://docs.python.org/3/library/configparser.html)** to manage training hyperparameters, backbone selections, data paths, and tracking backends cleanly:

- **Standard INI Configs**: [`configs/default_config.ini`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/default_config.ini), [`configs/fastai_convnext.ini`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/fastai_convnext.ini), [`configs/fastai_vit.ini`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/fastai_vit.ini), [`configs/skorch_resnet.ini`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/skorch_resnet.ini).
- **TOML Configs**: [`configs/config_advanced.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/config_advanced.toml).
- **JSON Configs**: [`configs/config_custom.json`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/configs/config_custom.json).
- 📖 **Guide**: [README_CONFIGPARSER.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_CONFIGPARSER.md).

---

## 📈 MLOps Experiment Tracking (W&B, MLflow, TensorBoard)

Unified tracker integration supporting:
1. **Weights & Biases (W&B)**: Real-time loss curves, validation accuracy, confusion matrix artifacts, and hyperparameter tables.
2. **MLflow**: Parameter and metric tracking via `mlflow.log_params` & `mlflow.log_metrics`.
3. **TensorBoard**: Interactive scalars and visual dashboards.
4. **Offline / Local Fallback**: Comprehensive JSON & CSV telemetry when offline without credentials.
- 📖 **Guide**: [README_TRACKING_WANDB.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite/README_TRACKING_WANDB.md).

---

## 🚀 Quick Start & Execution

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_fastai_suite

# 1. Create & Activate Virtual Environment
python -m venv venv_histology_fastai
.\venv_histology_fastai\Scripts\activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run fastai ConvNeXt-Base with configparser
python main_fastai.py --config configs/fastai_convnext.ini

# 4. Run fastai ViT-Base with TOML config
python main_fastai.py --config configs/config_advanced.toml

# 5. Run skorch ResNet50 (Scikit-Learn workflow)
python main_skorch.py --config configs/skorch_resnet.ini

# 6. Generate Comparison Leaderboard & Chart
python compare_suite.py
```

---

### 🐧 Ubuntu / WSL2 (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite

# 1. Create & Activate Virtual Environment
python3 -m venv venv_histology_fastai
source venv_histology_fastai/bin/activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run fastai Models
python3 main_fastai.py --config configs/fastai_convnext.ini
python3 main_fastai.py --config configs/fastai_vit.ini
python3 main_skorch.py --config configs/skorch_resnet.ini

# 4. Generate Leaderboard
python3 compare_suite.py
```

# Colorectal Histology Benchmark: Pure skorch + TOML Configuration + Weights & Biases (W&B) + Linter

A specialized benchmark suite engineered strictly around **[`skorch`](https://skorch.readthedocs.io/)** (`NeuralNetClassifier` wrapping PyTorch vision backbones with the Scikit-Learn API), modular **TOML configurations**, **[Weights & Biases (W&B)](https://wandb.ai/) MLOps tracking**, and automated **code quality / linting tooling** (`ruff`, `flake8`, `black`, `isort`, `mypy`).

---

## 🔬 Dataset & Multi-Class Clinical Scope

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

## 🏆 Model Families, TOML Profiles & skorch Pipelines

Each model family has its own dedicated TOML configuration and skorch `NeuralNetClassifier` pipeline:

| Model # | Foundation Architecture | skorch Backbone | Dedicated TOML Profile | Dedicated Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **ConvNeXt-Base** | `convnext_base` | [`configs/convnext_base.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/configs/convnext_base.toml) | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_CONVNEXT.md) |
| **2** | **Vision Transformer** | `vit_base_patch16_224` | [`configs/vit_base.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/configs/vit_base.toml) | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_VIT.md) |
| **3** | **EfficientNetV2** | `efficientnet_b3` | [`configs/efficientnet_v2.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/configs/efficientnet_v2.toml) | [README_EFFICIENTNET.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_EFFICIENTNET.md) |
| **4** | **Swin Transformer** | `swin_base_patch4_window7_224` | [`configs/swin_transformer.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/configs/swin_transformer.toml) | [README_SWIN.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_SWIN.md) |
| **5** | **ResNet50d** | `resnet50d` | [`configs/resnet50d.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/configs/resnet50d.toml) | [README_RESNET.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_RESNET.md) |

---

## 🛠️ Code Quality & Automated Linting

The repository includes a linter runner [`run_linter.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/run_linter.py) and [`pyproject.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/pyproject.toml) supporting:
- **Ruff**: Ultra-fast linting & import validation.
- **Flake8**: Strict PEP 8 style validation.
- **Black**: Deterministic formatting.
- **MyPy**: Static type verification.

To run the linter:
```bash
python run_linter.py
```
📖 **Guide**: [README_LINTER.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_LINTER.md)

---

## 📈 Weights & Biases (W&B) MLOps Integration

- **Live Callback**: Custom `SkorchWandbCallback` intercepting `on_epoch_end` to log epoch loss curves, learning rates, and validation accuracy.
- **Artifacts**: Automatically uploads normalized confusion matrix heatmaps and multi-class ROC curves to W&B.
- **Safe Offline Fallback**: Automatically falls back to `WANDB_MODE=offline` with local CSV/JSON telemetry when operating without API keys or internet access.
- 📖 **Guide**: [README_TOML_WANDB.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb/README_TOML_WANDB.md)

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_skorch_toml_wandb

# 1. Create & Activate Virtual Environment
python -m venv venv_skorch_toml
.\venv_skorch_toml\Scripts\activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Code Quality Linter
# python run_linter.py

# 4. Train Model 1 (ConvNeXt-Base) with TOML config + W&B
python main_skorch_wandb.py --config configs/convnext_base.toml

# 5. Train Model 2 (Vision Transformer)
python main_skorch_wandb.py --config configs/vit_base.toml

# 6. Train Model 5 (ResNet50d)
python main_skorch_wandb.py --config configs/resnet50d.toml

# 7. Generate 5-Model Comparison Leaderboard & Chart
python compare_skorch_models.py
```

---

### 🐧 Ubuntu / WSL2 (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb

# 1. Virtual Environment Setup
python3 -m venv venv_skorch_toml
source venv_skorch_toml/bin/activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt


# 3. Run Training
python3 main_skorch_wandb.py --config configs/convnext_base.toml
python3 main_skorch_wandb.py --config configs/resnet50d.toml
python3 compare_skorch_models.py
```

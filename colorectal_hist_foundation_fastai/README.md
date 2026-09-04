# Colorectal Histology Benchmark: Pathology Foundation Models in fastai + TOML + Weights & Biases (W&B) + Linter

A specialized benchmark suite engineered strictly around **State-of-the-Art Pathology Vision Foundation Models**, **[`fastai`](https://docs.fast.ai/)**, modular **TOML configurations**, **[Weights & Biases (W&B)](https://wandb.ai/) MLOps tracking**, and automated **code quality / linting tooling** (`ruff`, `flake8`, `black`, `isort`, `mypy`).

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

## 🏆 Pathology Foundation Model Families & fastai Pipelines

Each pathology foundation model has its own dedicated TOML configuration and fastai fine-tuning pipeline:

| Model # | Foundation Architecture | Backbone / Pre-training Source | Dedicated TOML Profile | Dedicated Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Owkin Phikon** | `owkin/phikon` (iBOT ViT-Base on 40M+ TCGA tiles) | [`configs/phikon.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/configs/phikon.toml) | [README_PHIKON.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_PHIKON.md) |
| **2** | **Paige Virchow** | `paige-ai/Virchow` (ViT-Huge on 1.5M WSIs) | [`configs/virchow.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/configs/virchow.toml) | [README_VIRCHOW.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_VIRCHOW.md) |
| **3** | **Harvard UNI** | `MahmoodLab/UNI` (ViT-Large on 100M+ patches) | [`configs/uni.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/configs/uni.toml) | [README_UNI.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_UNI.md) |
| **4** | **Meta DINOv2** | `facebook/dinov2-base` (Self-supervised ViT) | [`configs/dinov2.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/configs/dinov2.toml) | [README_DINOV2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_DINOV2.md) |
| **5** | **BiomedCLIP** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | [`configs/biomedclip.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/configs/biomedclip.toml) | [README_BIOMEDCLIP.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_BIOMEDCLIP.md) |

---

## 🛠️ Code Quality & Automated Linting

The repository includes a linter runner [`run_linter.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/run_linter.py) and [`pyproject.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/pyproject.toml) supporting:
- **Ruff**: Ultra-fast linting & import validation.
- **Flake8**: Strict PEP 8 style validation.
- **Black**: Deterministic formatting.
- **MyPy**: Static type verification.

To run the linter:
```bash
python run_linter.py
```
📖 **Guide**: [README_LINTER.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_LINTER.md)

---

## 📈 Weights & Biases (W&B) MLOps Integration

- **Live Callback**: Uses `fastai.callback.wandb.WandbCallback` to log batch losses, epoch metrics, and learning rates.
- **Artifacts**: Automatically uploads normalized confusion matrix heatmaps and multi-class ROC curves to W&B.
- **Safe Offline Fallback**: Automatically falls back to `WANDB_MODE=offline` with local CSV/JSON telemetry when operating without API keys or internet access.
- 📖 **Guide**: [README_TOML_WANDB.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai/README_TOML_WANDB.md)

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_pathology_foundation_fastai

# 1. Create & Activate Virtual Environment
python -m venv venv_foundation_fastai
.\venv_foundation_fastai\Scripts\activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Code Quality Linter
# python run_linter.py

# 4. Train Model 1 (Owkin Phikon) with TOML config + W&B
python main_fastai_foundation.py --config configs/phikon.toml

# 5. Train Model 2 (Paige Virchow)
python main_fastai_foundation.py --config configs/virchow.toml

# 6. Train Model 3 (Harvard UNI)
python main_fastai_foundation.py --config configs/uni.toml

# 7. Train Model 4 (Meta DINOv2)
python main_fastai_foundation.py --config configs/dinov2.toml

# 8. Generate 5-Model Comparison Leaderboard & Chart
python compare_foundation_models.py
```

---

### 🐧 Ubuntu / WSL2 (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai

# 1. Virtual Environment Setup
python3 -m venv venv_foundation_fastai
source venv_foundation_fastai/bin/activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Training
python3 main_fastai_foundation.py --config configs/phikon.toml
python3 main_fastai_foundation.py --config configs/dinov2.toml
python3 compare_foundation_models.py
```

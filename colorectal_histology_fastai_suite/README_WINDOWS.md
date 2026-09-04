# Running Colorectal Histology fastai & skorch Benchmark on Windows

This guide provides step-by-step instructions for creating an isolated Python virtual environment, configuring parameters via `configparser`, and running experiments on **Windows 10/11** using PowerShell or Command Prompt.

---

## 1. Virtual Environment Creation & Setup

Open **PowerShell** (or **CMD**) and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_fastai_suite

# 1. Create virtual environment
python -m venv venv_histology_fastai

# 2. Activate virtual environment
# In PowerShell:
.\venv_histology_fastai\Scripts\Activate.ps1

# (If script execution is restricted in PowerShell, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Or in Command Prompt (CMD):
.\venv_histology_fastai\Scripts\activate.bat

# 3. Upgrade pip and install all dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Running Models with `configparser` Configuration Profiles

### Option A: fastai with ConvNeXt-Base
```powershell
python main_fastai.py --config configs/fastai_convnext.ini
```

### Option B: fastai with Vision Transformer (ViT-Base)
```powershell
python main_fastai.py --config configs/fastai_vit.ini
```

### Option C: fastai with EfficientNetV2
```powershell
python main_fastai.py --config configs/fastai_efficientnet.ini
```

### Option D: skorch with ResNet50 (Scikit-Learn Interface)
```powershell
python main_skorch.py --config configs/skorch_resnet.ini
```

---

## 3. Dynamic CLI Hyperparameter Overrides

Any parameter in the INI/TOML/JSON config can be overridden on the command line:

```powershell
# Override epochs, batch size, and tracking backend:
python main_fastai.py --config configs/fastai_convnext.ini --epochs 10 --batch-size 32 --tracking-backend wandb
```

---

## 4. Leaderboard & Visual Metrics Comparison

```powershell
python compare_suite.py
```

Outputs:
- Formatted Markdown leaderboard printed to console.
- Multi-model comparison bar chart: `results/colorectal_histology_fastai_comparison.png`.
- Normalized confusion matrices: `results/confusion_matrix_fastai_*.png`.
- Multi-class ROC curves: `results/roc_curves_fastai_*.png`.

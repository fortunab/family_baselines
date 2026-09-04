# Windows Setup Guide: skorch + TOML + W&B Suite

This guide walks you through setting up an isolated virtual environment, running the linter, configuring TOML profiles, and logging experiments to **Weights & Biases (W&B)** on **Windows 10/11** via PowerShell or Command Prompt.

---

## 1. Virtual Environment Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_skorch_toml_wandb

# 1. Create virtual environment
python -m venv venv_skorch_toml

# 2. Activate virtual environment
# In PowerShell:
.\venv_skorch_toml\Scripts\Activate.ps1

# (If script execution is restricted: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Or in Command Prompt:
.\venv_skorch_toml\Scripts\activate.bat

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Code Quality & Linter Verification

```powershell
python run_linter.py
```

---

## 3. Weights & Biases Authentication (Optional)

If you have a W&B account, log in:
```powershell
wandb login
```
*(If no API key is provided, the tracker runs in safe offline mode with local CSV/JSON telemetry).*

---

## 4. Running skorch Models via Dedicated TOML Profiles

```powershell
# Model 1: ConvNeXt-Base
python main_skorch_wandb.py --config configs/convnext_base.toml

# Model 2: Vision Transformer (ViT-Base)
python main_skorch_wandb.py --config configs/vit_base.toml

# Model 3: EfficientNetV2
python main_skorch_wandb.py --config configs/efficientnet_v2.toml

# Model 4: Swin Transformer
python main_skorch_wandb.py --config configs/swin_transformer.toml

# Model 5: ResNet50d
python main_skorch_wandb.py --config configs/resnet50d.toml
```

---

## 5. Comparison Leaderboard

```powershell
python compare_skorch_models.py
```

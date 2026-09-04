# Windows Setup Guide: fastai Pathology Foundation Models

This guide walks you through setting up an isolated virtual environment, running the linter, configuring TOML profiles, and logging experiments to **Weights & Biases (W&B)** on **Windows 10/11** via PowerShell or Command Prompt.

---

## 1. Virtual Environment Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_pathology_foundation_fastai

# 1. Create virtual environment
python -m venv venv_foundation_fastai

# 2. Activate virtual environment
# In PowerShell:
.\venv_foundation_fastai\Scripts\Activate.ps1

# (If script execution is restricted: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Or in Command Prompt:
.\venv_foundation_fastai\Scripts\activate.bat

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

## 4. Running fastai Foundation Models via Dedicated TOML Profiles

```powershell
# Model 1: Owkin Phikon (iBOT ViT-Base)
python main_fastai_foundation.py --config configs/phikon.toml

# Model 2: Paige Virchow (ViT-Huge Whole-Slide Foundation)
python main_fastai_foundation.py --config configs/virchow.toml

# Model 3: Harvard UNI (ViT-Large Foundation)
python main_fastai_foundation.py --config configs/uni.toml

# Model 4: Meta DINOv2 (Vision Foundation Backbone)
python main_fastai_foundation.py --config configs/dinov2.toml

# Model 5: Microsoft BiomedCLIP
python main_fastai_foundation.py --config configs/biomedclip.toml
```

---

## 5. Comparison Leaderboard

```powershell
python compare_foundation_models.py
```

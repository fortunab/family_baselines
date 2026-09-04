# Windows 10/11 Setup Guide: Herlev Cytology Pathology Foundation Suite

Step-by-step instructions for configuring and running the **Herlev Cervical Cytology Pathology Foundation fastai Suite** on Windows 10 and 11 using PowerShell or CMD.

---

## 1. Prerequisites

- Python 3.10, 3.11, or 3.12 (64-bit)
- NVIDIA GPU with CUDA 11.8 or 12.x (Recommended, CPU fallback supported)
- Windows PowerShell with Execution Policy enabled:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 2. Virtual Environment Setup

From the project root:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_pathology_foundation_fastai

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip, wheel, and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install required dependencies
pip install -r requirements.txt
```

---

## 3. Verify Code Quality with Linter

Run the integrated multi-linter script:
```powershell
python run_linter.py
```
This runs:
1. AST syntax tree verification across all Python scripts
2. `ruff` style and error checking
3. `flake8` compliance
4. `black` code format validation

---

## 4. Running Foundation Models

Train any of the 5 supported foundation models:

```powershell
# 1. Owkin Phikon (Pan-Cancer Histology Foundation ViT)
python main_herlev_fastai.py --config configs/phikon.toml

# 2. Paige Virchow (632M parameter ViT-Huge Whole Slide Foundation Model)
python main_herlev_fastai.py --config configs/virchow.toml

# 3. Harvard UNI (ViT-Large 100M+ patch model)
python main_herlev_fastai.py --config configs/uni.toml

# 4. Meta DINOv2 (Self-Supervised Vision Foundation Model)
python main_herlev_fastai.py --config configs/dinov2.toml

# 5. Microsoft BiomedCLIP (Biomedical Vision Backbone)
python main_herlev_fastai.py --config configs/biomedclip.toml
```

---

## 5. Weights & Biases (W&B) Windows Setup

To stream telemetry live to your W&B dashboard:
```powershell
$env:WANDB_API_KEY = "your_actual_wandb_api_key_here"
python main_herlev_fastai.py --config configs/phikon.toml
```

If you do not have an API key or prefer local operation:
```powershell
# Suite will automatically run in offline mode and record all metrics to results/
python main_herlev_fastai.py --config configs/phikon.toml --no_wandb
```

---

## 6. Generate Leaderboard and Plots

```powershell
python compare_herlev_models.py
```
Generated outputs in `results/`:
- `herlev_foundation_benchmark_summary.csv`
- `herlev_foundation_comparison.png`
- `confusion_matrix_<backbone>.png`
- `roc_curves_<backbone>.png`

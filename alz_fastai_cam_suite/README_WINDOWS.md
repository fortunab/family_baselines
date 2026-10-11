# Alzheimer MRI fastai CAM Suite: Windows 11 Guide

Complete setup, execution, and troubleshooting guide for **Windows 11** using PowerShell and Command Prompt.

---

## 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: 3.11.x (Recommended: 3.11.9)
- **PowerShell**: Version 5.1+ or PowerShell 7+
- **Hardware**: CPU (Intel/AMD) or NVIDIA GPU with CUDA 11.8/12.1+

---

## 2. Environment Setup

### Option A: Standard `venv` (PowerShell)
```powershell
# Navigate to project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\alzheimer_mri_fastai_cam_suite

# Run the automated PowerShell setup script
.\scripts\setup_venv.ps1

# Or manual creation:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PowerShell Execution Policy**: If you receive a script execution error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Option B: Conda / Miniconda
```powershell
# Create environment from environment.yml
conda env create -f environment.yml

# Activate
conda activate alzheimer-cam
```

---

## 3. Running Heatmaps and Metric Calculations

### 1-Line Execution (Single Command)
Generate all heatmaps (Grad-CAM, Grad-CAM++, Score-CAM, Consensus, Variance) and compute Step 1 & Step 2 metrics:
```powershell
python visualize_cam_heatmaps.py
```

### Full Training & Classification Evaluation (Step 1)
```powershell
python train_and_evaluate.py --epochs 3 --lr 0.001
```

### Quantitative CAM Benchmark (Step 2 Comparison)
```powershell
python compare_cam_metrics.py --samples-per-class 5
```

---

## 4. Code Quality & Linters
Run the 4-stage quality audit on Windows:
```powershell
python run_linter.py
```

Or individual tools:
```powershell
python -m ruff check .
python -m flake8 --max-line-length=120 .
python -m black --check --line-length=120 .
```

---

## 5. Generated Artifacts on Windows
All visual outputs and quantitative reports are saved locally:
- `results\heatmaps\cam_breakdown_NonDemented.png`
- `results\heatmaps\cam_breakdown_VeryMildDemented.png`
- `results\heatmaps\cam_breakdown_MildDemented.png`
- `results\heatmaps\cam_breakdown_ModerateDemented.png`
- `results\heatmaps\dementia_progression_gallery.png`
- `results\heatmaps\cam_metrics_comparison.png`
- `results\metrics\step_metrics_summary.json`

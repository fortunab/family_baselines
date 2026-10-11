# Alzheimer MRI Grayscale Benchmark: Windows 11 Guide

Instructions for configuring and running the suite on **Windows 11** using PowerShell and Command Prompt.

---

## 1. System Requirements
- Windows 10 or 11 (64-bit)
- Python 3.11.x (Installed via Python.org, pyenv-win, or winget)
- PowerShell 5.1+ / PowerShell 7+

---

## 2. Environment Setup

### Option A: Standard `venv` (PowerShell)
```powershell
# Navigate to directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\alzheimer_mri_grayscale_benchmark_suite

# Run automated PowerShell setup
.\scripts\setup_venv.ps1

# Or manual execution:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **PowerShell Execution Policy**: If script execution is restricted, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Option B: Conda / Miniconda
```powershell
.\scripts\setup_conda.ps1
conda activate alzheimer-grayscale-bench
```

---

## 3. Running Experiments on Windows

### Full Pipeline (1-Line Execution)
```powershell
python run_full_benchmark_suite.py
```

### Grayscale Preprocessing Step
```powershell
python preprocess_grayscale_dataset.py
```

### Classical ML Family Experiment
```powershell
python run_ml_family_experiment.py
```

### Foundational Vision Backbone Comparison
```powershell
python run_foundational_comparison.py
```

### Quality Audit
```powershell
python run_linter.py
```

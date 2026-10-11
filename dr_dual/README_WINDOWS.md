# Windows 11 / PowerShell Execution Guide

This guide details configuring and executing the Dual-Database Clinical Machine Learning Framework natively on Windows 11.

---

## 1. Prerequisites

- Windows 10/11 (64-bit)
- PowerShell 5.1 or PowerShell 7+
- Python 3.10+ (Recommended: Python 3.11.9)
- Visual C++ Redistributable 2015–2022

Verify Python:
```powershell
python --version
```

---

## 2. Environment Setup

### Option A: Automated PowerShell Setup Script
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_venv.ps1
```

### Option B: Manual Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Running the Benchmarks

### Step 1: Synthesize and Harmonize Dual Databases
```powershell
python generate_dual_databases.py --dr-samples-per-db 250 --isic-samples-per-db 150
```

### Step 2: Run Diabetic Retinopathy ML Family Benchmark
```powershell
python run_ml_family_benchmark.py
```
This extracts 42 radiomic biomarker features, trains and benchmarks 9 ML algorithms, evaluates cross-database generalization gaps, and generates publication diagnostic figures.

### Step 3: Run Dermatology Dual-Database Benchmark (ISIC + HAM10000)
```powershell
python run_isic_ham_benchmark.py
```

### Step 4: Run Static Quality Audit (Linter)
```powershell
python run_linter.py
```

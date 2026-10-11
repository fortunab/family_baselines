# Python `venv` Environment Guide: Alzheimer MRI Grayscale Benchmark

Instructions for using Python standard library `venv`.

---

## 1. Setup Instructions

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / WSL (Bash)
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Pinned Requirements
Includes:
- `torch>=2.2.0`, `torchvision>=0.17.0`, `fastai>=2.7.14`
- `scikit-learn>=1.4.0`, `scipy>=1.12.0`, `numpy>=1.26.0`, `pandas>=2.2.0`
- `pillow>=10.2.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`
- `pyyaml>=6.0.1`, `timm>=0.9.16`
- `ruff>=0.4.0`, `flake8>=7.0.0`, `black>=24.3.0`

---

## 3. Running Scripts
```bash
python run_full_benchmark_suite.py
```

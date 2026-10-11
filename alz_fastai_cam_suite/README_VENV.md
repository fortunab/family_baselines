# Python `venv` Setup Guide: Alzheimer MRI CAM Suite

Instructions for managing dependencies with the Python standard library `venv`.

---

## 1. Setup Instructions

### On Windows (PowerShell)
```powershell
# 1. Create a dedicated virtual environment
python -m venv venv

# 2. Activate the environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### On Linux / macOS / WSL
```bash
# 1. Create virtual environment
python3.11 -m venv venv

# 2. Activate
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Pinned Requirements (`requirements.txt`)
- `torch>=2.2.0`, `torchvision>=0.17.0`
- `fastai>=2.7.14`, `fastcore>=1.5.29`
- `scikit-learn>=1.4.0`
- `scipy>=1.12.0`
- `pillow>=10.2.0`
- `matplotlib>=3.8.0`
- `pandas>=2.2.0`
- `pyyaml>=6.0.1`
- `ruff>=0.4.0`, `flake8>=7.0.0`, `black>=24.3.0`

---

## 3. Verifying Installation
```bash
python -c "import torch, fastai, sklearn, matplotlib; print('Environment initialized successfully!')"
```

---

## 4. Deactivating
To deactivate the virtual environment when finished:
```bash
deactivate
```

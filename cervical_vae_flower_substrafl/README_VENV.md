# Standard Python venv Setup & Execution Guide

This guide details configuring and executing Suite 27 (`cervical_cells_vae_flower_substra`) using standard Python virtual environments (`venv`) and `requirements.txt`.

---

## 🚀 Virtual Environment Creation

### Using Setup Script
```powershell
# Windows PowerShell
.\scripts\setup_venv.ps1

# Linux / WSL
bash ./scripts/setup_venv.sh
```

### Manual venv Setup
```powershell
# 1. Create virtual environment
python -m venv venv_cervical_vae

# 2. Activate environment
# On Windows:
.\venv_cervical_vae\Scripts\Activate.ps1
# On Linux/macOS:
source venv_cervical_vae/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Execution Commands

```powershell
# Flower FedAvg with fastai & VAE
python main_federated_flower.py --framework fastai --preprocessing vae --rounds 5

# Substra FedProx with skorch & VAE
python main_federated_substra.py --framework skorch --preprocessing vae --strategy FedProx

# Preprocessing Reconstruction Benchmark
python compare_cervical_vae.py

# Quality Gate
python run_linter.py
```

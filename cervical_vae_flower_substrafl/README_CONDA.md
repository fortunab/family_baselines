# Conda Environment Setup & Execution Guide

This guide details setting up Suite 27 (`cervical_cells_vae_flower_substra`) using Anaconda or Miniconda via `environment.yml`.

---

## 🚀 Environment Creation

### Using Setup Script
```powershell
# Windows PowerShell
.\scripts\setup_conda.ps1

# Linux / WSL
bash ./scripts/setup_conda.sh
```

### Manual Conda Creation
```bash
conda env create -f environment.yml
conda activate cervical_cells_vae_fs
```

---

## 🏃 Execution Commands

Once activated:

```powershell
# 1. Flower Federated Learning (fastai, VAE)
python main_federated_flower.py --framework fastai --preprocessing vae --rounds 5

# 2. Substra Federated Learning (skorch, VAE)
python main_federated_substra.py --framework skorch --preprocessing vae --strategy FedProx

# 3. Direct Cytology Representation Benchmark (Raw vs VAE)
python compare_cervical_vae.py --subsample 50 --epochs 3

# 4. Static Code Quality Suite (AST, Ruff, Flake8, Black)
python run_linter.py
```

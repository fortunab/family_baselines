# Conda / Mamba Environment Guide for skorch Federated Learning

This document describes how to configure and execute the **Flower Federated Learning Suite for Cervical Cytology with skorch** using Conda or Mamba with `environment.yml`.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_conda.ps1
```

### Linux / WSL2 Bash
```bash
chmod +x setup_conda.sh
./setup_conda.sh
```

---

## 2. Manual Step-by-Step Configuration

### Step 2.1: Create Environment from `environment.yml`
```bash
conda env create -f environment.yml
```
*Note: If you have `mamba` installed, replace `conda` with `mamba` for faster dependency solving:*
```bash
mamba env create -f environment.yml
```

### Step 2.2: Activate Environment
```bash
conda activate cervical_cells_skorch
```

### Step 2.3: Verify GPU Availability
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 3. Running Experiments

```bash
# Fast Dry Run
python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb

# Multi-Strategy Comparison
python compare_federated_strategies.py
```

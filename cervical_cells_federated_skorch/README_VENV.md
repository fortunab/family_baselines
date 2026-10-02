# Standard Virtual Environment (venv) Guide for skorch Federated Learning

This document describes how to configure and execute the **Flower Federated Learning Suite for Cervical Cytology with skorch** using Python's built-in `venv` environment manager.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_venv.ps1
```

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 2. Manual Step-by-Step Configuration

### Step 2.1: Create Virtual Environment
```bash
# Windows
python -m venv venv_cervical_skorch

# Linux / macOS
python3 -m venv venv_cervical_skorch
```

### Step 2.2: Activate Environment
```bash
# Windows PowerShell:
.\venv_cervical_skorch\Scripts\Activate.ps1

# Linux / WSL2 / macOS:
source venv_cervical_skorch/bin/activate
```

### Step 2.3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. Running Federated Experiments

### Fast Verification Run
```powershell
python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

### Full Benchmark Run (FedAvg, 5 Clients, ConvNeXt)
```powershell
python main_federated_skorch.py --config configs/default.toml
```

### Benchmark Strategy Comparison
```powershell
python compare_federated_strategies.py
```

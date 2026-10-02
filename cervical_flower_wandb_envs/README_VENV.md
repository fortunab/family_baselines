# Standard Virtual Environment (venv) Setup & Execution Guide

This document describes how to configure and execute the **Flower Federated Learning Suite for Cervical Cytology** using Python's built-in `venv` environment manager.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
# From project root:
.\setup_venv.ps1
```

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 2. Manual Step-by-Step Configuration

### Step 2.1: Verify Python Version
Flower and fastai in this suite support Python 3.10, 3.11, and 3.12 (CUDA 12.x wheels).
```bash
python --version
```

### Step 2.2: Create the Virtual Environment
```bash
# Windows
python -m venv venv_cervical_flower

# Linux / macOS
python3 -m venv venv_cervical_flower
```

### Step 2.3: Activate the Environment
```bash
# Windows PowerShell:
.\venv_cervical_flower\Scripts\Activate.ps1

# Windows CMD:
.\venv_cervical_flower\Scripts\activate.bat

# Linux / WSL2 / macOS:
source venv_cervical_flower/bin/activate
```

### Step 2.4: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

*(Optional PyTorch CUDA 12.x explicit wheel if not detected):*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Running Federated Experiments

### Fast Verification Run (Dry Run)
```powershell
python main_federated_flower.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

### Full Benchmark Run (FedAvg, 5 Clients, ConvNeXt)
```powershell
python main_federated_flower.py --config configs/convnext.toml
```

### Non-IID Dirichlet Skew with FedProx ($\mu = 1.0$)
```powershell
python main_federated_flower.py --config configs/fedprox_non_iid.toml
```

### Benchmark Strategy Comparison
```powershell
python compare_federated_strategies.py --strategies FedAvg FedProx FedAdam --num_rounds 5
```

---

## 4. Code Quality & Linting
Verify formatting and syntax compliance across all project files:
```powershell
python run_linter.py
```

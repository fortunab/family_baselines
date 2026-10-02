# Pyenv Environment Guide

[Pyenv](https://github.com/pyenv/pyenv) (and [pyenv-win](https://github.com/pyenv-win/pyenv-win)) allows you to manage multiple isolated Python versions across projects.

This project pins Python `3.11.9` in `.python-version`.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_pyenv.ps1
```

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_pyenv.sh
./setup_pyenv.sh
```

---

## 2. Manual Configuration

### Step 2.1: Install Target Python Version
```bash
pyenv install 3.11.9
```

### Step 2.2: Set Local Version for this Project
```bash
pyenv local 3.11.9
```
*Verify with:*
```bash
python --version
# Output: Python 3.11.9
```

### Step 2.3: Create and Activate Virtualenv
```bash
# Using standard venv under pyenv python:
python -m venv venv_cervical_flower

# Activate:
# Windows PowerShell:
.\venv_cervical_flower\Scripts\Activate.ps1
# Linux / macOS:
source venv_cervical_flower/bin/activate

# Install requirements:
pip install -r requirements.txt
```

---

## 3. Running Experiments
```bash
python main_federated_flower.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

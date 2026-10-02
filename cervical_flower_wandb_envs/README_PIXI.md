# Pixi Environment & Workflow Guide

[Pixi](https://pixi.sh/) is a high-performance, cross-platform package management and task running tool built on the Conda ecosystem with deterministic lockfiles.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_pixi.ps1
```

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_pixi.sh
./setup_pixi.sh
```

---

## 2. Pixi Tasks & Workflow

`pixi.toml` defines predefined tasks for running simulations, linter checks, and comparisons directly without manually activating environments.

### Install Pixi (if not already installed)
```powershell
# Windows PowerShell:
iwr -useb https://pixi.sh/install.ps1 | iex

# Linux / macOS:
curl -fsSL https://pixi.sh/install.sh | bash
```

### Install Project Dependencies
```bash
pixi install
```

### Running Pixi Tasks
```bash
# Fast Dry Run
pixi run dry-run

# Run Federated Training (FedAvg with ConvNeXt)
pixi run train

# Run FedProx Non-IID Dirichlet Skew
pixi run train-fedprox

# Compare FL Strategies
pixi run compare

# Run Linter Quality Suite
pixi run lint
```

### Interactive Shell
```bash
pixi shell
python main_federated_flower.py --help
```

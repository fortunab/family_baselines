# Pixi Package Manager Guide: FedML Cervical Cytology

Pixi is a fast, reproducible conda-forge and PyPI package manager built on top of `rattler` and `rip`.

---

## 1. Prerequisites & Installation

Install Pixi on Windows PowerShell:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

Or on Linux / WSL:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

---

## 2. Automated Bootstrap Script

We provide ready-to-run setup scripts:
- **Windows**:
  ```powershell
  .\scripts\setup_pixi.ps1
  ```
- **Linux / WSL**:
  ```bash
  bash scripts/setup_pixi.sh
  ```

---

## 3. Manifest Overview (`pixi.toml`)

The workspace specifies dual platforms (`win-64`, `linux-64`) and combines conda-forge binaries with PyPI wheels:
```toml
[workspace]
name = "cervical_cells_federated_fedml"
channels = ["conda-forge"]
platforms = ["win-64", "linux-64"]

[dependencies]
python = ">=3.11,<3.12"
numpy = ">=1.24"
pandas = ">=2.0"
scikit-learn = ">=1.3"
ruff = ">=0.1.0"
black = ">=23.0.0"
flake8 = ">=6.0.0"

[pypi-dependencies]
torch = ">=2.0.0"
torchvision = ">=0.15.0"
fastai = ">=2.7.12"
skorch = ">=0.15.0"
timm = ">=0.9.0"
wandb = ">=0.16.0"
```

---

## 4. Running Pixi Tasks

Pixi tasks defined in `pixi.toml` allow one-command execution:
```powershell
# Run fastai FedML test
pixi run test-fastai

# Run skorch FedML test
pixi run test-skorch

# Run strategy comparison
pixi run compare

# Run static quality checks
pixi run lint
```

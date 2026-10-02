# Pixi Environment Guide: Substra Cervical Cytology

Modern, cross-platform reproducibility using **Pixi** for Windows (`win-64`) and Linux (`linux-64`).

---

## 1. Automated Bootstrap

Run the self-bootstrapping script which verifies the Pixi binary (installing it if missing) and installs all dependencies:

### Windows PowerShell:
```powershell
.\setup_pixi.ps1
```

### Linux / Ubuntu / WSL:
```bash
chmod +x setup_pixi.sh
./setup_pixi.sh
```

---

## 2. Predefined Pixi Tasks

Execute tasks directly with `pixi run`:

```powershell
# 1. Full Substra federated training with fastai
pixi run train-fastai

# 2. Full Substra federated training with skorch
pixi run train-skorch

# 3. Fast 2-round dry run with Random Best Select
pixi run dry-run

# 4. Multi-strategy benchmark (FedAvg vs FedProx)
pixi run benchmark

# 5. Static code quality checker (AST, Ruff, Flake8, Black)
pixi run lint
```

---

## 3. Lockfile Reproducibility

The repository includes a fully solved [`pixi.lock`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/pixi.lock) guaranteeing byte-exact package versions across machines.

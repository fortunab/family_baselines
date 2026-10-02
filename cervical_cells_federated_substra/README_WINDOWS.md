# Windows PowerShell Quickstart Guide: Substra Cervical Cytology

Recommended setup and execution commands on native Windows 11 / Windows 10 with PowerShell.

---

## 1. Environment Choice

Choose your preferred manager:
```powershell
# Option A: Pixi (Recommended)
.\setup_pixi.ps1

# Option B: uv
.\setup_uv.ps1

# Option C: Conda
.\setup_conda.ps1

# Option D: venv
.\setup_venv.ps1
```

---

## 2. Launching Substra Simulation

```powershell
# Run fastai engine
python main_federated_substra.py --config configs/default.toml --framework fastai

# Run skorch engine
python main_federated_substra.py --config configs/default.toml --framework skorch

# Dry run with Random Best Select
python main_federated_substra.py --config configs/default.toml --subsample 60 --num_rounds 2 --no_wandb
```

# uv Environment Guide: Substra Cervical Cytology

Extremely fast package resolution and virtual environment management via **Astral uv**.

---

## 1. Automated Setup

### Windows PowerShell:
```powershell
.\setup_uv.ps1
```

### Linux / Ubuntu / WSL:
```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

---

## 2. Running Workflows via `uv run`

```powershell
# Run with fastai
uv run python main_federated_substra.py --config configs/default.toml --framework fastai

# Run with skorch
uv run python main_federated_substra.py --config configs/default.toml --framework skorch

# Benchmark comparison
uv run python compare_substra_strategies.py --rounds 3

# Static linter suite
uv run python run_linter.py
```

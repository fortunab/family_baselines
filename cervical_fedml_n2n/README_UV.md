# uv Environment Guide

`uv` is an extremely fast Python package and project manager written in Rust by Astral.

---

## 🚀 Setup & Execution

### Windows PowerShell
```powershell
.\scripts\setup_uv.ps1
```

### Linux / WSL / macOS
```bash
bash scripts/setup_uv.sh
```

---

## 🏃 Running Commands with uv

```powershell
# Run federated experiment directly with uv run
uv run python main_federated_fedml.py --preprocessing n2n_cyclegan

# Run linter
uv run python run_linter.py
```

# Standard Python venv Guide

This guide details configuring a lightweight standard Python virtual environment.

---

## 🚀 Setup & Activation

### Windows PowerShell
```powershell
.\scripts\setup_venv.ps1
# Or manually:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / WSL / macOS
```bash
bash scripts/setup_venv.sh
# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Pyenv Version Management Guide

This project locks the exact Python interpreter runtime to **Python 3.11.9** via `.python-version`.

---

## 🚀 Setup & Execution

### Windows (pyenv-win)
```powershell
.\scripts\setup_pyenv.ps1
```

### Linux / WSL / macOS
```bash
bash scripts/setup_pyenv.sh
```

---

## 🔍 Verification

```bash
python --version
# Expected output: Python 3.11.9
```

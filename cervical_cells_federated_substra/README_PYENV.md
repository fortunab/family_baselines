# Pyenv Guide: Substra Cervical Cytology

Python version isolation pinning Python `3.11.9` via `.python-version`.

---

## 1. Automated Setup

### Windows PowerShell:
```powershell
.\setup_pyenv.ps1
```

### Linux / Ubuntu / WSL:
```bash
chmod +x setup_pyenv.sh
./setup_pyenv.sh
```

---

## 2. Activation and Execution

```powershell
.\.venv_pyenv\Scripts\Activate.ps1
python main_federated_substra.py --config configs/default.toml --framework fastai
python main_federated_substra.py --config configs/default.toml --framework skorch
```

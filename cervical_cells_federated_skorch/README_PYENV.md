# Pyenv Multi-Version Guide for skorch Federated Learning

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

```bash
# Install target Python:
pyenv install 3.11.9
pyenv local 3.11.9

# Create virtualenv:
python -m venv venv_cervical_skorch
source venv_cervical_skorch/bin/activate  # Or .\venv_cervical_skorch\Scripts\Activate.ps1
pip install -r requirements.txt
```

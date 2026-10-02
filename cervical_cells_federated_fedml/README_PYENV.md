# Pyenv Environment Guide: FedML Cervical Cytology

This guide covers setting up Python version isolation using `pyenv` (or `pyenv-win` on Windows) targeting Python 3.11.9.

---

## 1. Prerequisites

- **Windows**: Install `pyenv-win` via PowerShell:
  ```powershell
  Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" | Invoke-Expression
  ```
- **Linux / WSL**: Install standard `pyenv`:
  ```bash
  curl https://pyenv.run | bash
  ```

---

## 2. Automated Bootstrap Script

- **Windows**:
  ```powershell
  .\scripts\setup_pyenv.ps1
  ```
- **Linux / WSL**:
  ```bash
  bash scripts/setup_pyenv.sh
  ```

---

## 3. Manual Steps

1. Install Python 3.11.9:
   ```bash
   pyenv install 3.11.9
   ```

2. Pin the directory version to 3.11.9:
   ```bash
   pyenv local 3.11.9
   ```

3. Create and activate a dedicated virtual environment:
   ```bash
   python -m venv venv_pyenv_fedml
   source venv_pyenv_fedml/bin/activate  # Or .\venv_pyenv_fedml\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

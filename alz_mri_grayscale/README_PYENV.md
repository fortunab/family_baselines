# `pyenv` Setup Guide: Alzheimer MRI Grayscale Benchmark

Instructions for managing Python versions and isolated environments using `pyenv` and `pyenv-win`.

---

## 1. Local Python Pinning
This repository declares the exact Python version in `.python-version`:
```
3.11.9
```

---

## 2. Windows (`pyenv-win`)
```powershell
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3. Linux / WSL (`pyenv`)
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Execution
```bash
python run_full_benchmark_suite.py
```

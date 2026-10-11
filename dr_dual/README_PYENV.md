# Pyenv Environment Guide

This document describes how to pin and manage Python 3.11.9 with `pyenv`.

---

## 1. Pinning Python 3.11.9

The root `.python-version` file pins the interpreter version:
```bash
pyenv install 3.11.9
pyenv local 3.11.9
python --version  # Output: Python 3.11.9
```

---

## 2. Virtual Environment Creation

```bash
python -m venv .venv
source .venv/bin/activate  # (or .\.venv\Scripts\Activate.ps1 on Windows)
pip install -r requirements.txt
```

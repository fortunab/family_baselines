# Virtual Environment (`venv`) Guide

This document describes how to create, isolate, and manage a standard Python `venv` for this project.

---

## 1. Creation

```bash
# Windows
python -m venv venv

# Linux / WSL
python3 -m venv venv
```

---

## 2. Activation

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / macOS / WSL
source venv/bin/activate
```

---

## 3. Dependency Installation

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4. Deactivation

```bash
deactivate
```

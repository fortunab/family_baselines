# uv Environment Guide: FedML Cervical Cytology

`uv` is an extremely fast Python package manager and resolver written in Rust by Astral.

---

## 1. Prerequisites & Installation

Install `uv` on Windows PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or on Linux / macOS / WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Automated Bootstrap Script

- **Windows**:
  ```powershell
  .\scripts\setup_uv.ps1
  ```
- **Linux / WSL**:
  ```bash
  bash scripts/setup_uv.sh
  ```

---

## 3. Manual Step-by-Step Setup

1. **Create Virtual Environment**:
   ```bash
   uv venv --python 3.11 .venv
   ```

2. **Activate Environment**:
   - Windows PowerShell:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / WSL:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

---

## 4. Running Experiments with uv

Execute commands directly within the isolated environment:
```powershell
# Run with fastai
python main_federated_fedml.py --framework fastai

# Run with skorch
python main_federated_fedml.py --framework skorch

# Verify code quality
python run_linter.py
```

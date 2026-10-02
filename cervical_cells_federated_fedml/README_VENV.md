# Standard venv Environment Guide: FedML Cervical Cytology

This guide covers creating a standard lightweight Python `venv` virtual environment using standard `pip`.

---

## 1. Automated Bootstrap Script

- **Windows PowerShell**:
  ```powershell
  .\scripts\setup_venv.ps1
  ```
- **Linux / WSL**:
  ```bash
  bash scripts/setup_venv.sh
  ```

---

## 2. Manual Step-by-Step Instructions

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv_federated_fedml
   ```

2. **Activate the Environment**:
   - Windows PowerShell:
     ```powershell
     .\venv_federated_fedml\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     .\venv_federated_fedml\Scripts\activate.bat
     ```
   - Linux / macOS / WSL:
     ```bash
     source venv_federated_fedml/bin/activate
     ```

3. **Upgrade Core Tools & Install Dependencies**:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

---

## 3. Running Workflows

```powershell
# Run FedML with fastai
python main_federated_fedml.py --framework fastai

# Run FedML with skorch
python main_federated_fedml.py --framework skorch

# Execute strategy benchmark
python compare_fedml_strategies.py
```

# Standard Python venv Guide: Substra Cervical Cytology

Native virtual environment using standard Python `venv` and `pip`.

---

## 1. Automated Setup

### Windows PowerShell:
```powershell
.\setup_venv.ps1
```

### Linux / Ubuntu / WSL:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 2. Activation and Execution

### Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
python main_federated_substra.py --config configs/default.toml --framework fastai
python main_federated_substra.py --config configs/default.toml --framework skorch
python compare_substra_strategies.py --rounds 3
python run_linter.py
```

### Linux / WSL:
```bash
source .venv/bin/activate
python main_federated_substra.py --config configs/default.toml --framework fastai
python main_federated_substra.py --config configs/default.toml --framework skorch
```

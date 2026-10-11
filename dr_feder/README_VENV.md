# Python Virtual Environments (venv) Reference Guide
## Federated Diabetic Retinopathy Suite

Ghid tehnic pentru crearea și gestionarea mediilor virtuale Python native (`venv`).

---

## 1. Comenzi Rapide

### Windows PowerShell
```powershell
python -m venv venv_dr_fedml
.\venv_dr_fedml\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

### Linux / WSL / Bash
```bash
python3 -m venv venv_dr_fedml
source venv_dr_fedml/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

# Python Virtual Environments (venv) Reference Guide
## Skin Lesion Federated Learning Suite

Acest ghid oferă toate detaliile tehnice privind utilizarea mediilor virtuale Python native (`venv`) pe Windows și Linux.

---

## 1. De Ce `venv`?
- **Izolare Totală**: Previne conflictele între versiunile pachetelor globale (PyTorch, W&B, scikit-learn).
- **Fără Dependențe Externe**: `venv` face parte din biblioteca standard Python (începând cu Python 3.3).
- **Ușor și Rapid**: Consum minim de disc și pornire instantanee.

---

## 2. Comenzi Rapide: Creare, Activare, Dezactivare

### Pe Windows (PowerShell)
```powershell
# 1. Creare
python -m venv venv_fedml_derm

# 2. Activare
.\venv_fedml_derm\Scripts\Activate.ps1

# 3. Instalare dependente
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Dezactivare (cand termini lucrul)
deactivate
```

### Pe Linux / macOS / WSL (Bash)
```bash
# 1. Creare
python3 -m venv venv_fedml_derm

# 2. Activare
source venv_fedml_derm/bin/activate

# 3. Instalare dependente
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Dezactivare
deactivate
```

---

## 3. Depanare Probleme Comune (Troubleshooting)

### A. Eroare `ExecutionPolicy` pe Windows PowerShell
Dacă primești eroarea:
```
File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```
**Soluție**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### B. Pachete Lipsă sau Versiuni Conflictuale
Dacă întâmpini erori la import:
```bash
# Reinstalează pachetele în mod curat
pip install --force-reinstall -r requirements.txt
```

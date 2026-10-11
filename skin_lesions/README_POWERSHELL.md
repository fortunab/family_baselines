# Windows PowerShell Setup & Execution Guide
## Federated Learning Skin Lesion Suite (FedML / W&B / TOML / INI)

Acest ghid oferă instrucțiuni pas-cu-pas pentru instalarea, configurarea și rularea suitei federate de clasificare a leziunilor dermatologice pe **Windows (PowerShell)**, folosind fie **venv (Python standard)**, fie **Conda / Mamba**.

---

## Opțiunea A: Configurare cu `venv` (Recomandat pentru medii standard)

### Pasul 1: Deschide PowerShell ca Utilizator Normal
Apasă `Win + X` și selectează **Terminal** sau **Windows PowerShell**. Navighează în folderul proiectului:
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\skin_lesion_federated_wandb_suite
```

### Pasul 2: Permisiuni pentru Scripturi (dacă este necesar)
Dacă execuția de scripturi este blocată (`ExecutionPolicy`), rulează o singură dată:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Pasul 3: Crearea Mediului Virtual `venv`
Asigură-te că folosești Python 3.10, 3.11 sau 3.12:
```powershell
python --version
python -m venv venv_fedml_derm
```

### Pasul 4: Activarea Mediului Virtual
```powershell
.\venv_fedml_derm\Scripts\Activate.ps1
```
*(Promptul PowerShell va afișa prefixul `(venv_fedml_derm)`).*

### Pasul 5: Actualizare Pip și Instalare Dependențe
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Opțiunea B: Configurare cu `Conda` / `Miniconda`

### Pasul 1: Crearea Mediului Conda din `environment.yml`
```powershell
conda env create -f environment.yml
```
*Alternativ, crearea manuală:*
```powershell
conda create -n skin_lesion_fedml python=3.11 -y
conda activate skin_lesion_fedml
pip install -r requirements.txt
```

### Pasul 2: Activarea Mediului Conda
```powershell
conda activate skin_lesion_fedml
```

---

## Pasul 6: Verificarea Integrității Codului (Linter & Tests)

Rulează suita de teste automate pentru a verifica încărcătoarele TOML/INI, partiționarea Dirichlet și agregarea FedProx:
```powershell
python run_tests.py
```

Rulează auditul static complet (AST, Ruff, Flake8, Black):
```powershell
python run_linter.py
```

---

## Pasul 7: Rularea Experimentelor Federate

### 1. Rulare Implicită cu Configurație TOML (FedProx, 15 Runde, Mod Offline W&B)
```powershell
python run_federated_benchmark.py --config configs/federated_config.toml
```

### 2. Rulare cu Configurație INI (`configparser`)
```powershell
python run_federated_benchmark.py --config configs/federated_config.ini
```

### 3. Compararea Algoritmilor Federate (FedAvg vs. FedProx)

**Rulare FedAvg standard:**
```powershell
python run_federated_benchmark.py --algorithm fedavg --rounds 15 --mu 0.0
```

**Rulare FedProx (Regularizare împotriva Non-IID Drift):**
```powershell
python run_federated_benchmark.py --algorithm fedprox --rounds 15 --mu 0.01 --alpha 0.5
```

### 4. Activare Sincronizare Online Weights & Biases (Opțional)
Dacă dorești să transmiți metricile live pe tabloul de bord [wandb.ai](https://wandb.ai):
```powershell
# Autentificare în W&B (doar prima dată)
wandb login

# Rulare cu mod online
python run_federated_benchmark.py --wandb-mode online
```

---

## Pasul 8: Localizarea Rezultatelor și Figurilor Generate

Toate metricile și graficele se salvează automat în folderul `results/`:
- `results/figures/federated_convergence_curves.png`: Curbele de învățare (pierdere, acuratețe, sensibilitate referire).
- `results/figures/non_iid_dirichlet_distribution.png`: Distribuția etichetelor pe spitalele clinice.
- `results/figures/dermoscopy_confusion_matrix.png`: Matricea de confuzie cu cele 7 clase dermatoscopice.
- `results/figures/malignancy_screening_roc.png`: Curba ROC pentru triajul leziunilor maligne (Melanom, BCC, AKIEC).
- `results/figures/dermoscopy_7class_gallery.png`: Galeria morfologică de leziuni.
- `results/metrics/federated_benchmark_summary.json`: Export complet structurat al metricilor.
- `results/models/`: Puncte de salvare ale modelului optim global (`.pt`).

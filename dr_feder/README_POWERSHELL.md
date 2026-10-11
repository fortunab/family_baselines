# Windows PowerShell Setup & Execution Guide
## Federated Diabetic Retinopathy Suite (FedML / W&B / TOML / INI)

Acest ghid oferă instrucțiuni pas-cu-pas pentru instalarea, configurarea și rularea suitei federate de clasificare a retinopatiei diabetice (5 stadii ICDR) pe **Windows (PowerShell)**, folosind fie **venv (Python standard)**, fie **Conda / Mamba**.

---

## Opțiunea A: Configurare cu `venv` (Recomandat)

### Pasul 1: Deschide PowerShell și Navighează în Folder
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\diabetic_retinopathy_federated_wandb_suite
```

### Pasul 2: Permisiuni pentru Scripturi (dacă este necesar)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Pasul 3: Crearea Mediului Virtual `venv`
```powershell
python -m venv venv_dr_fedml
```

### Pasul 4: Activarea Mediului Virtual
```powershell
.\venv_dr_fedml\Scripts\Activate.ps1
```

### Pasul 5: Instalarea Dependențelor
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Opțiunea B: Configurare cu `Conda` / `Miniconda`

```powershell
# Creare mediu din environment.yml
conda env create -f environment.yml

# Activare
conda activate dr_fedml_suite
```

---

## Pasul 6: Verificare Integritate și Teste

```powershell
# Rulare suita de 8 teste unitare si de integrare
python run_tests.py

# Rulare audit static (AST, Ruff, Flake8, Black)
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

### 3. Comparare FedAvg vs. FedProx

**Rulare FedAvg:**
```powershell
python run_federated_benchmark.py --algorithm fedavg --rounds 15 --mu 0.0
```

**Rulare FedProx (Regularizare împotriva Non-IID Drift):**
```powershell
python run_federated_benchmark.py --algorithm fedprox --rounds 15 --mu 0.01 --alpha 0.5
```

### 4. Sincronizare Online cu Weights & Biases (Opțional)
```powershell
wandb login
python run_federated_benchmark.py --wandb-mode online
```

---

## Pasul 8: Vizualizarea Rezultatelor

Fișierele generate se găsesc în:
- `results/figures/federated_convergence_curves.png`: Curbele de învățare (pierdere, QWK, sensibilitate referire).
- `results/figures/non_iid_dirichlet_distribution.png`: Alocarea cazurilor Non-IID pe cele 4 centre oftalmologice.
- `results/figures/dr_5grade_confusion_matrix.png`: Matricea de confuzie cu cele 5 stadii ICDR și limita de referire.
- `results/figures/dr_referral_roc_curve.png`: Curba ROC pentru triajul retinopatiei ce pune în pericol vederea (Grad >= 2).
- `results/figures/fundus_5grade_gallery.png`: Galeria de imagini fundus de ochi pentru fiecare stadiu.
- `results/metrics/federated_dr_benchmark_summary.json`: Export JSON cantitativ complet.
- `results/models/`: Puncte de salvare PyTorch (`.pt`).

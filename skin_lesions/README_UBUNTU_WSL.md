# Linux / Ubuntu / WSL2 Setup & Execution Guide
## Federated Learning Skin Lesion Suite (FedML / W&B / TOML / INI)

Acest ghid oferă instrucțiuni pas-cu-pas pentru instalarea, configurarea și rularea suitei federate pe distribuții **Linux (Ubuntu 20.04/22.04/24.04, Debian) și Windows Subsystem for Linux (WSL2)**, folosind fie **venv (Python standard)**, fie **Conda / Mamba**.

---

## Pregătire Inițială Sistem (Ubuntu / Debian / WSL2)

Deschide terminalul bash și instalează pachetele de bază:
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git build-essential
```

Navighează în folderul proiectului:
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/skin_lesion_federated_wandb_suite
# Sau dacă lucrezi direct în directorul Linux home:
# cd ~/skin_lesion_federated_wandb_suite
```

---

## Opțiunea A: Configurare cu `venv` (Recomandat pe Linux)

### Pasul 1: Crearea Mediului Virtual
```bash
python3 -m venv venv_fedml_derm
```

### Pasul 2: Activarea Mediului Virtual
```bash
source venv_fedml_derm/bin/activate
```
*(Terminalul va afișa prefixul `(venv_fedml_derm)`).*

### Pasul 3: Instalarea Dependențelor
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Opțiunea B: Configurare cu `Conda` / `Miniconda` / `Mamba`

### Pasul 1: Crearea Mediului din Fișierul `environment.yml`
```bash
conda env create -f environment.yml
```
*Alternativ, crearea manuală directă:*
```bash
conda create -n skin_lesion_fedml python=3.11 -y
conda activate skin_lesion_fedml
pip install -r requirements.txt
```

### Pasul 2: Activarea Mediului Conda
```bash
conda activate skin_lesion_fedml
```

---

## Pasul 4: Rularea Verificării Integrității Codului

Rulează suita de teste automate:
```bash
python run_tests.py
```

Rulează auditul static linter (AST, Ruff, Flake8, Black):
```bash
python run_linter.py
```

---

## Pasul 5: Rularea Experimentelor Federate

### 1. Rulare Implicită cu Configurație TOML (FedProx, 15 Runde, W&B Offline)
```bash
python run_federated_benchmark.py --config configs/federated_config.toml
```

### 2. Rulare cu Configurație INI (`configparser`)
```bash
python run_federated_benchmark.py --config configs/federated_config.ini
```

### 3. Compararea Algoritmilor Federate (FedAvg vs. FedProx)

**Rulare FedAvg standard:**
```bash
python run_federated_benchmark.py --algorithm fedavg --rounds 15 --mu 0.0
```

**Rulare FedProx (Regularizare împotriva Non-IID Drift):**
```bash
python run_federated_benchmark.py --algorithm fedprox --rounds 15 --mu 0.01 --alpha 0.5
```

### 4. Suport GPU CUDA (Dacă este disponibil)
Dacă ai un GPU NVIDIA cu CUDA instalat pe Linux sau WSL2, poți specifica dispozitivul în comandă:
```bash
python run_federated_benchmark.py --config configs/federated_config.toml
# Codul detectează automat GPU dacă este setat `device = "cuda"` în toml/ini.
```

### 5. Sincronizare Online cu Weights & Biases
```bash
# Autentificare o singură dată
wandb login

# Rulare cu logging live în cloud
python run_federated_benchmark.py --wandb-mode online
```

---

## Pasul 6: Vizualizarea Rezultatelor

Fișierele generate se găsesc în:
- `results/figures/federated_convergence_curves.png`
- `results/figures/non_iid_dirichlet_distribution.png`
- `results/figures/dermoscopy_confusion_matrix.png`
- `results/figures/malignancy_screening_roc.png`
- `results/figures/dermoscopy_7class_gallery.png`
- `results/metrics/federated_benchmark_summary.json`
- `results/models/` (checkpoint-urile modelului global salvate cu PyTorch)

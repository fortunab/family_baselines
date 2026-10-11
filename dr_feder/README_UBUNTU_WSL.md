# Linux / Ubuntu / WSL2 Setup & Execution Guide
## Federated Diabetic Retinopathy Suite (FedML / W&B / TOML / INI)

Acest ghid oferă instrucțiuni pas-cu-pas pentru instalarea, configurarea și rularea suitei federate pe distribuții **Linux (Ubuntu 20.04/22.04/24.04, Debian) și Windows Subsystem for Linux (WSL2)**, folosind fie **venv (Python standard)**, fie **Conda / Mamba**.

---

## Pregătire Sistem

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git build-essential
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/diabetic_retinopathy_federated_wandb_suite
```

---

## Opțiunea A: Configurare cu `venv`

```bash
# 1. Creare mediu virtual
python3 -m venv venv_dr_fedml

# 2. Activare
source venv_dr_fedml/bin/activate

# 3. Instalare dependente
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Opțiunea B: Configurare cu `Conda` / `Miniconda`

```bash
conda env create -f environment.yml
conda activate dr_fedml_suite
```

---

## Pasul 4: Teste și Verificare Calitate

```bash
# Rulare teste unitare si de integrare
python run_tests.py

# Rulare audit static (AST, Ruff, Flake8, Black)
python run_linter.py
```

---

## Pasul 5: Rularea Experimentelor Federate

### 1. Rulare Implicită cu Configurație TOML (FedProx, 15 Runde, Mod Offline W&B)
```bash
python run_federated_benchmark.py --config configs/federated_config.toml
```

### 2. Rulare cu Configurație INI (`configparser`)
```bash
python run_federated_benchmark.py --config configs/federated_config.ini
```

### 3. Comparare FedAvg vs. FedProx

**Rulare FedAvg:**
```bash
python run_federated_benchmark.py --algorithm fedavg --rounds 15 --mu 0.0
```

**Rulare FedProx:**
```bash
python run_federated_benchmark.py --algorithm fedprox --rounds 15 --mu 0.01 --alpha 0.5
```

### 4. Sincronizare Cloud Weights & Biases (Opțional)
```bash
wandb login
python run_federated_benchmark.py --wandb-mode online
```

# Federated Learning Clinical Ophthalmology Suite
## 5-Grade Diabetic Retinopathy Classification with FedML, FedProx, Weights & Biases, TOML & INI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Offline%20%26%20Online-gold.svg)](https://wandb.ai/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff & Flake8](https://img.shields.io/badge/linter-ruff%20%7C%20flake8-green.svg)](https://github.com/astral-sh/ruff)

O suită completă de cercetare și implementare clinică pentru **învățare automată federată (Federated Learning)** aplicată stadializării retinopatiei diabetice (5 clase ICDR) și triajului pentru referire oftalmologică, simulând cooperarea între 4 centre oftalmologice internaționale (EyePACS California, Messidor-2 Paris, IDRiD India, APTOS Asia) cu distribuție **Non-IID Dirichlet**.

---

## 1. Caracteristici Cheie ale Proiectului

1. **Stadializare Retiniană în 5 Stadii ICDR**:
   - `Grade 0 (No_DR)`: Fundus normal, maculă clară, fără microanevrisme.
   - `Grade 1 (Mild_NPDR)`: Doar microanevrisme izolate.
   - `Grade 2 (Moderate_NPDR)`: Hemoragii punctiforme, exudate dure, pete vătuate (cotton wool spots).
   - `Grade 3 (Severe_NPDR)`: Regula 4-2-1 (hemoragii severe în 4 cadrane, dilatații venoase în 2, anomalii microvasculare IRMA în 1).
   - `Grade 4 (Proliferative_DR)`: Neovascularizație pe disc/retină, hemoragie preretiniană/vitreeană.
2. **Criteriu Clinic de Referire Vitală**:
   - Stadiile $\ge 2$ (Moderat, Sever, Proliferativ) impun referire la medicul specialist oftalmolog pentru prevenirea pierderii vederii.
   - Sensibilitate de referire țintă: $\ge 95.0\%$.
3. **Optimizare Federată FedProx împotriva Client Drift**:
   - Termen proximal $\mu \frac{1}{2}\|w - w_t\|^2$ pentru atenuarea deviației cauzate de camerele diferite (Topcon vs Canon vs Zeiss) și prevalența Non-IID Dirichlet ($\alpha = 0.5$).
4. **Integrare Weights & Biases (W&B)**:
   - Mod **Offline** implicit (complet autonom) și mod **Online** pentru cloud.
   - Înregistrare Quadratic Weighted Kappa ($\kappa_w$), acuratețe, sensibilitate referire și salvare artefacte de imagini.
5. **Configurare Duală TOML și INI (`configparser`)**:
   - Suport atât pentru `configs/federated_config.toml`, cât și pentru `configs/federated_config.ini`.
6. **Audit Static și Testare Automată**:
   - 0 erori la validarea AST, Ruff, Flake8 și Black.
   - Suită completă de 8 teste unitare și de integrare (`run_tests.py`).

---

## 2. Structura Proiectului

```
diabetic_retinopathy_federated_wandb_suite/
├── configs/
│   ├── federated_config.toml      # Configurație completă în format TOML
│   └── federated_config.ini       # Configurație completă în format INI (configparser)
├── data/
│   ├── manifests/                 # Manifeste JSON și metadata CSV
│   └── raw/images/                # Imagini fundus de ochi 3-canale RGB (MD5 verified)
├── results/
│   ├── figures/                   # Diagrame de convergență, matrice de confuzie, ROC
│   ├── metrics/                   # Sumar JSON al metricilor federate
│   └── models/                    # Puncte de salvare PyTorch (.pt)
├── src/
│   ├── __init__.py
│   ├── client.py                  # Client federat (antrenare locală fundus, FedProx)
│   ├── config_loader.py           # Încărcător unificat TOML / INI (configparser)
│   ├── dataset_generator.py       # Generator și manager fundus 5 clase ICDR
│   ├── evaluator.py               # Metricile clinice (QWK, Acuratețe, Sensibilitate, ECE)
│   ├── federated_partitioner.py   # Partiționer Non-IID Dirichlet Dir(alpha)
│   ├── models.py                  # RetinaResNet și operatori de divergență
│   ├── server.py                  # Server federat (agregare FedAvg/FedProx, test global)
│   ├── visualizer.py              # Generare figuri de publicație
│   └── wandb_logger.py            # Integrare W&B (mod offline & online)
├── tests/
│   ├── __init__.py
│   └── test_federated_dr_pipeline.py # Suită teste unitare & integrare
├── environment.yml                # Specificație mediu Conda / Mamba
├── pyproject.toml                 # Configurații build, Ruff și Black
├── requirements.txt               # Dependențe pip
├── run_federated_benchmark.py     # Script principal CLI de rulare a experimentului
├── run_linter.py                  # Audit static de calitate (AST, Ruff, Flake8, Black)
├── run_tests.py                   # Rulare teste fără dependențe externe
├── README.md                      # Acest ghid principal
├── README_POWERSHELL.md           # Ghid pas-cu-pas detaliat pentru Windows PowerShell
├── README_UBUNTU_WSL.md           # Ghid pas-cu-pas detaliat pentru Linux / WSL2
├── README_VENV.md                 # Ghid aprofundat medii virtuale (venv)
├── README_CONDA.md                # Ghid aprofundat Conda / Mamba
└── README_FEDML_WANDB.md          # Ghid teoretic FedML, FedProx și W&B
```

---

## 3. Ghid Rapid de Instalare și Rulare

### Windows (PowerShell)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\diabetic_retinopathy_federated_wandb_suite
python -m venv venv_dr_fedml
.\venv_dr_fedml\Scripts\Activate.ps1
pip install -r requirements.txt
python run_tests.py
python run_linter.py
python run_federated_benchmark.py --config configs/federated_config.toml
```

### Linux / Ubuntu / WSL2 (Bash)
```bash
cd diabetic_retinopathy_federated_wandb_suite
python3 -m venv venv_dr_fedml
source venv_dr_fedml/bin/activate
pip install -r requirements.txt
python run_tests.py
python run_linter.py
python run_federated_benchmark.py --config configs/federated_config.toml
```

### Conda / Mamba
```bash
conda env create -f environment.yml
conda activate dr_fedml_suite
python run_federated_benchmark.py --config configs/federated_config.ini
```

---

## 4. Vizualizări Generate

Rularea generează automat în `results/figures/`:
1. `federated_convergence_curves.png`: Curbele de învățare (pierdere, QWK, sensibilitate referire pe parcursul rundelor).
2. `non_iid_dirichlet_distribution.png`: Alocarea Non-IID a celor 5 stadii pe cele 4 centre clinice.
3. `dr_5grade_confusion_matrix.png`: Matricea de confuzie 5x5 cu limita clinică de referire.
4. `dr_referral_roc_curve.png`: Curba ROC pentru triajul retinopatiei ce pune în pericol vederea (Grad $\ge 2$).
5. `fundus_5grade_gallery.png`: Galeria morfologică cu scanări de fundus de ochi pentru fiecare stadiu.

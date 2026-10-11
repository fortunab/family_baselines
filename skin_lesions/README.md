# Federated Learning Clinical Dermatology Suite
## 7-Class Skin Lesion Classification with FedML, FedProx, Weights & Biases, TOML & INI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Offline%20%26%20Online-gold.svg)](https://wandb.ai/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff & Flake8](https://img.shields.io/badge/linter-ruff%20%7C%20flake8-green.svg)](https://github.com/astral-sh/ruff)

O suită completă de cercetare și implementare clinică pentru **învățare automată federată (Federated Learning)** aplicată diagnosticului dermatoscopic al leziunilor cutanate (melanom și leziuni benigne), simulând colaborarea multi-instituțională între 4 spitale internaționale cu distribuție **Non-IID Dirichlet**.

---

## 1. Caracteristici Cheie ale Proiectului

1. **Clasificare Dermatoscopică în 7 Clase (HAM10000 / ISIC Benchmark)**:
   - `MEL`: Melanom (Malign, Urgență Înaltă de Excizie)
   - `NV`: Nevi Melanocitari (Benign)
   - `BCC`: Carcinom Bazocelular (Malign)
   - `AKIEC`: Keratoze Actinice / Boala Bowen (Pre-malign / Malign)
   - `BKL`: Keratoze Benigne (Lentigine solare, keratoze seboreice)
   - `DF`: Dermatofibrom (Benign)
   - `VASC`: Leziuni Vasculare (Angioame, hemangioame)
2. **Arhitectură Federată Robustă (FedML Core)**:
   - Simulare server-client multi-instituțional (Spitalul Viena, Centrul Sydney, Clinica MSK New York, Spitalul Charité Berlin).
   - Suport nativ pentru **FedAvg** (McMahan et al.) și **FedProx** (Li et al., cu termen de regularizare proximală $\mu \frac{1}{2}\|w - w_t\|^2$ pentru atenuarea deviației cauzate de heterogenitatea Non-IID).
   - Partiționare Dirichlet Non-IID $\text{Dir}(\alpha)$ cu $\alpha = 0.5$ pentru a reflecta prevalența clinică reală din diferite centre geografice.
3. **Integrare Completă Weights & Biases (W&B)**:
   - Suport dublu mod: **Offline** (`WANDB_MODE=offline`) complet autonom (fără cont sau internet) și **Online** pentru tablouri de bord cloud în timp real.
   - Înregistrarea traiectoriei fiecărei runde: pierdere globală, divergență clienți, sensibilitate referire malignitate, metrici ECE și salvare artefacte de imagini.
4. **Sistem Hibrid de Configurare (TOML & INI / configparser)**:
   - `configs/federated_config.toml`: Format modern parsabil prin `tomllib` (Python 3.11+) sau `tomli`.
   - `configs/federated_config.ini`: Format standard compatibil 100% cu `configparser` din biblioteca standard Python.
   - `src/config_loader.py`: Manager unificat ce suportă ambele formate și validare strictă prin dataclass tipizat.
5. **Calitate și Audit Static de Cod**:
   - 0 erori la validarea AST, Ruff, Flake8 și Black.
   - Suită completă de teste unitare și de integrare (`run_tests.py`).

---

## 2. Structura Folderului de Proiect

```
skin_lesion_federated_wandb_suite/
├── configs/
│   ├── federated_config.toml      # Configurație completă în format TOML
│   └── federated_config.ini       # Configurație completă în format INI (configparser)
├── data/
│   ├── manifests/                 # Manifeste JSON și metadata CSV (HAM10000)
│   └── raw/images/                # Imagini dermatoscopice 3-canale RGB (MD5 verified)
├── results/
│   ├── figures/                   # Diagrame de convergență, matrice de confuzie, ROC
│   ├── metrics/                   # Sumar cantitativ JSON al rundelor federate
│   └── models/                    # Puncte de salvare PyTorch (.pt) ale modelului optim
├── src/
│   ├── __init__.py
│   ├── abcd_biomarker_extractor.py # Extracție algoritmică a regulii Stolz ABCD (A, B, C, D, TDS)
│   ├── advanced_metrics.py        # Metrice clinice (Wilson CIs, pAUC 80%, Brier, DCA, Echitate)
│   ├── cam_explainer.py           # Explicabilitate spațială Grad-CAM și Grad-CAM++
│   ├── client.py                  # Client federat (antrenare locală, termen FedProx)
│   ├── clinical_report_generator.py # Generator de rapoarte clinice structurate (standard EDF/IDS)
│   ├── config_loader.py           # Încărcător unificat TOML / INI (configparser)
│   ├── dataset_generator.py       # Generator și manager dermatoscopie 7 clase HAM10000
│   ├── evaluator.py               # Metricile clinice (Acuratețe, Kappa, Sensibilitate, ECE)
│   ├── federated_partitioner.py   # Partiționer Non-IID Dirichlet Dir(alpha)
│   ├── models.py                  # SkinLesionResNet și operatori de divergență
│   ├── multimodal_fusion.py       # Fuziune multimodală (imagine dermatoscopică + metadate pacient)
│   ├── server.py                  # Server federat (agregare FedAvg/FedProx, test global)
│   ├── visualizer.py              # Generare figuri de publicație (DCA, Echitate, ABCD, Grad-CAM++)
│   └── wandb_logger.py            # Integrare W&B (mod offline & online)
├── tests/
│   ├── __init__.py
│   ├── test_advanced_skin.py      # Teste unitare pentru metrice avansate, ABCD, DCA, Grad-CAM++
│   └── test_federated_pipeline.py # Suită teste unitare & integrare federată
├── results/
│   ├── figures/                   # Diagrame convergență, DCA, Fitzpatrick, ABCD, Grad-CAM++
│   ├── metrics/                   # Sumar cantitativ JSON al rundelor federate și activităților avansate
│   ├── models/                    # Puncte de salvare PyTorch (.pt) ale modelului optim
│   └── reports/                   # Rapoarte clinice structurate dermatologice (.md și .json)
├── environment.yml                # Specificație mediu Conda / Mamba
├── pyproject.toml                 # Configurații build, Ruff și Black
├── requirements.txt               # Dependențe pip
├── run_advanced_skin_suite.py     # Script master pentru rularea activităților clinice avansate
├── run_federated_benchmark.py     # Script principal CLI de rulare a experimentului federat
├── run_linter.py                  # Audit static de calitate a codului (AST, Ruff, Flake8, Black)
├── run_tests.py                   # Rulare teste fără dependențe externe (15/15 teste)
├── README.md                      # Acest ghid principal
├── README_POWERSHELL.md           # Ghid pas-cu-pas detaliat pentru Windows PowerShell
├── README_UBUNTU_WSL.md           # Ghid pas-cu-pas detaliat pentru Linux / WSL2
├── README_VENV.md                 # Ghid aprofundat medii virtuale (venv)
├── README_CONDA.md                # Ghid aprofundat Conda / Mamba
└── README_FEDML_WANDB.md          # Ghid teoretic FedML, FedProx și W&B
```

---

## 3. Ghid Rapid de Instalare și Rulare

### A. Pe Windows (PowerShell)
```powershell
# 1. Navigare în director
cd C:\Users\Lenovo\.gemini\antigravity\scratch\skin_lesion_federated_wandb_suite

# 2. Creare și activare venv
python -m venv venv_fedml_derm
.\venv_fedml_derm\Scripts\Activate.ps1

# 3. Instalare dependențe
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Rulare audit calitate și teste
python run_tests.py
python run_linter.py

# 5. Rulare experiment federat (FedProx, 15 runde)
python run_federated_benchmark.py --config configs/federated_config.toml

# 6. Rulare activități clinice avansate (DCA, Echitate Fitzpatrick, ABCD, Grad-CAM++, Rapoarte)
python run_advanced_skin_suite.py
```

### B. Pe Linux / Ubuntu / WSL2 (Bash)
```bash
# 1. Navigare în director
cd ~/skin_lesion_federated_wandb_suite  # sau calea corespunzătoare

# 2. Creare și activare venv
python3 -m venv venv_fedml_derm
source venv_fedml_derm/bin/activate

# 3. Instalare dependențe
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Rulare audit calitate și teste
python run_tests.py
python run_linter.py

# 5. Rulare experiment federat
python run_federated_benchmark.py --config configs/federated_config.toml

# 6. Rulare activități clinice avansate
python run_advanced_skin_suite.py
```

### C. Cu Conda / Mamba (Toate Platformele)
```bash
conda env create -f environment.yml
conda activate skin_lesion_fedml
python run_federated_benchmark.py --config configs/federated_config.toml
python run_advanced_skin_suite.py
```

---

## 4. Ghid de Configurare (TOML vs. INI)

Sistemul suportă ambele formate de configurare în mod interschimbabil:

| Parametru | Semnificație Clinică / Tehnică | Valoare Implicită |
|---|---|:---:|
| `algorithm` | Algoritmul federat (`fedprox`, `fedavg`, `centralized`) | `fedprox` |
| `rounds` | Numărul de runde de comunicare server-client | `15` |
| `local_epochs` | Epoci locale de antrenare per spital pe rundă | `2` |
| `dirichlet_alpha` | Concentrația Dirichlet Non-IID (valoare mică = eterogenitate mare) | `0.5` |
| `fedprox_mu` | Pondere termen proximal FedProx împotriva divergenței | `0.01` |
| `mode` (W&B) | Mod de înregistrare W&B (`offline`, `online`, `disabled`) | `offline` |

Pentru a rula cu fișierul INI citit prin `configparser`:
```bash
python run_federated_benchmark.py --config configs/federated_config.ini
```

---

## 5. Metricile Clinice Evaluat

1. **Acuratețe 7 Clase & Balanced Accuracy**: Evaluarea pe cohorta de test globală independentă (84 scanări).
2. **Cohen's Kappa ($\kappa$)**: Măsura acordului ordinal/nominal ajustată pentru predicția la întâmplare.
3. **Sensibilitate Referire Malignitate (Melanom + BCC + AKIEC)**: Măsură critică pentru siguranța pacientului. Prag clinic obligatoriu: $\ge 90\%$.
4. **Specificitate Referire**: Previne supraîncărcarea clinicilor de dermatologie chirurgicală.
5. **Expected Calibration Error (ECE)**: Calibrarea probabilistică a modelului la predicția confidenței clinice.

---

## 6. Vizualizări de Publicație Generate

Rularea runner-ului generează automat următoarele figuri în `results/figures/`:
1. `federated_convergence_curves.png`: Evoluția pierderii, acurateței și sensibilității pe parcursul celor 15 runde.
2. `non_iid_dirichlet_distribution.png`: Alocarea cazurilor Non-IID pe cele 4 spitale participante.
3. `dermoscopy_confusion_matrix.png`: Matricea de confuzie 7x7 cu adnotări clinice.
4. `malignancy_screening_roc.png`: Curba ROC pentru triajul leziunilor maligne vs. benigne.
5. `dermoscopy_7class_gallery.png`: Panou demonstrativ cu exemple dermatoscopice din fiecare clasă.

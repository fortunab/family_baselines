# Cervical Cytology Federated Learning Suite (Flower + fastai + Multi-Env)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flower 1.39+](https://img.shields.io/badge/Flower-1.39.0-orange.svg)](https://flower.ai/)
[![fastai 2.7+](https://img.shields.io/badge/fastai-2.7.12-green.svg)](https://docs.fast.ai/)
[![pixi](https://img.shields.io/badge/pixi-package%20manager-yellow.svg)](https://pixi.sh/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A medical-grade, publication-ready Federated Learning (FL) benchmark suite for **7-class cervical cytology single-cell Pap smear classification (Herlev dataset)**. This project integrates the **Flower (`flwr`)** framework with **fastai / PyTorch** vision backbones, modular **TOML** configurations, centralized holdout evaluation, non-IID clinical partitioning, **Weights & Biases (W&B)** experiment telemetry, and **first-class multi-environment management** (`venv`, `conda`, `pixi`, `pyenv`, `uv`).

---

## 🔬 Clinical Context (Herlev Pap Smear Cytology)

Cervical cancer is one of the most preventable gynecological malignancies when detected at precancerous dysplasia stages via routine Pap smear screening.

In multi-center screening programs:
- Cervical cytology slides collected at regional screening clinics and pathology laboratories cannot be aggregated into a centralized repository due to patient privacy laws (GDPR, HIPAA).
- **Flower Federated Learning** enables distributed cytology clinics to collaboratively train a global vision classifier without transmitting patient cell images across clinics. Only mathematical weight updates are communicated to the coordinating Flower server.

### 7 Cervical Cytology Cell Classes (Herlev Taxonomy)
1. **01_normal_superficiel**: Normal superficial squamous epithelial cells (benign)
2. **02_normal_intermediate**: Normal intermediate squamous epithelial cells (benign)
3. **03_normal_columnar**: Normal endocervical columnar epithelial cells (benign)
4. **04_light_dysplastic**: Mild squamous dysplasia (CIN 1 / Low-grade SIL)
5. **05_moderate_dysplastic**: Moderate squamous dysplasia (CIN 2 / High-grade SIL)
6. **06_severe_dysplastic**: Severe squamous dysplasia (CIN 3 / High-grade SIL)
7. **07_carcinoma_in_situ**: Carcinoma in situ / invasive cervical squamous cell carcinoma

---

## 📦 Multi-Environment Tooling (venv, conda, pixi, pyenv, uv)

This benchmark suite natively supports all five standard Python environment paradigms:

| Environment Manager | Config File | One-Click Setup (Windows) | One-Click Setup (Linux/macOS) |
|:---|:---|:---|:---|
| **Python venv** | `requirements.txt` | `.\setup_venv.ps1` | `bash setup_venv.sh` |
| **Conda / Mamba** | `environment.yml` | `.\setup_conda.ps1` | `bash setup_conda.sh` |
| **Pixi** | `pixi.toml` | `.\setup_pixi.ps1` | `bash setup_pixi.sh` |
| **Pyenv** | `.python-version` | `.\setup_pyenv.ps1` | `bash setup_pyenv.sh` |
| **Astral uv** | `pyproject.toml` | `.\setup_uv.ps1` | `bash setup_uv.sh` |

---

## ✨ Key Features

1. **Flower (`flwr`) NumPyClient Architecture**:
   - `FlwrFastaiCervicalClient` subclasses `flwr.client.NumPyClient` to interface directly with `fastai.Learner`.
   - Parameter serialization between NumPy arrays and PyTorch model state dicts.

2. **Non-IID & IID Clinical Partitioning**:
   - Simulates regional screening clinic heterogeneity using symmetric/asymmetric Dirichlet distributions ($\text{Dir}(\alpha)$).
   - Separates an **unseen 15% holdout test set** evaluated by the server callback after every round.

3. **Federated Optimization Strategies**:
   - **FedAvg** (McMahan et al.): Standard federated parameter averaging.
   - **FedProx** (Li et al.): Proximal regularization penalty $\frac{\mu}{2} \|w - w_t\|^2$ preventing client drift under extreme non-IID conditions.
   - **FedAdam** (Reddi et al.): Adaptive server-side optimization with first and second momentum tracking.

4. **SOTA Vision Backbones (timm, torchvision, foundation models)**:
   - ConvNeXt (`convnext_small`)
   - Vision Transformer (`vit_base_patch16_224`)
   - EfficientNetV2 (`tf_efficientnetv2_m`)
   - Swin Transformer (`swin_base_patch4_window7_224`)
   - ResNet50d (`resnet50d`)
   - Owkin Phikon pathology foundation model support

5. **MLOps, Reproducibility & Telemetry**:
   - TOML configurations (`configs/default.toml`, etc.) with full CLI overrides.
   - Weights & Biases round-by-round metric tracking and artifact logging with offline fallback.
   - Automated strategy benchmark runner (`compare_federated_strategies.py`).
   - Unified linter runner (`run_linter.py`) verifying AST, Ruff, Flake8, and Black formatting.

---

## 📁 Repository Structure

```
cervical_cells_federated_flower/
├── configs/
│   ├── default.toml             # Default baseline (ConvNeXt, FedAvg, 5 clinics)
│   ├── convnext.toml            # ConvNeXt-Small configuration
│   ├── vit.toml                 # Vision Transformer (ViT-Base-224)
│   ├── efficientnet.toml        # EfficientNetV2-M
│   ├── swin.toml                # Swin Transformer
│   ├── resnet.toml              # ResNet50d baseline
│   ├── phikon.toml              # Pathology Foundation model
│   └── fedprox_non_iid.toml     # Non-IID Dirichlet (alpha=0.5) with FedProx (mu=1.0)
├── src/
│   ├── __init__.py              # OpenMP & fastcore patches
│   ├── toml_config.py           # TOML loader & CLI parser
│   ├── wandb_tracker.py         # Weights & Biases and CSV telemetry logger
│   ├── dataset.py               # 7-class loader, 15% holdout, Dirichlet partitioning
│   ├── fastai_engine.py         # Model factory, parameter serialization, fastai local trainer
│   ├── flower_client.py         # Flower NumPyClient implementation & client factory
│   ├── flower_server.py         # Flower strategy builder & server evaluation callback
│   └── evaluator.py             # Multiclass metrics: Acc, BalAcc, F1, ROC-AUC, CM
├── main_federated_flower.py     # Main Federated Learning orchestration entrypoint
├── compare_federated_strategies.py # FedAvg vs. FedProx vs. FedAdam benchmark runner
├── run_linter.py                # AST, Ruff, Flake8, Black validation suite
├── requirements.txt             # Pip dependency specifications
├── environment.yml              # Conda / Mamba environment definition
├── pixi.toml                    # Pixi package manager configuration
├── .python-version              # Pyenv Python version file
├── pyproject.toml               # Build system & Astral uv configuration
├── setup_venv.ps1 / .sh         # Virtual environment setup script
├── setup_conda.ps1 / .sh        # Conda environment setup script
├── setup_pixi.ps1 / .sh         # Pixi environment setup script
├── setup_pyenv.ps1 / .sh        # Pyenv environment setup script
├── setup_uv.ps1 / .sh           # Astral uv setup script
├── README.md                    # Primary project documentation
├── README_VENV.md               # Python venv guide
├── README_CONDA.md              # Conda & Mamba guide
├── README_PIXI.md               # Pixi package manager guide
├── README_PYENV.md              # Pyenv guide
├── README_UV.md                 # Astral uv guide
├── README_WINDOWS.md            # Windows 10/11 PowerShell guide
├── README_UBUNTU_WSL.md         # Ubuntu / Debian / WSL2 guide
├── README_FLOWER.md             # Flower framework deep dive
├── README_FEDPROX.md            # Non-IID cytology & FedProx analysis
├── README_TOML_WANDB.md         # Configuration & telemetry documentation
└── README_LINTER.md             # Code quality and style guide
```

---

## 🚀 Quickstart

### 1. Choose Your Environment

#### Option A: Python venv
```bash
python -m venv venv_flower_cervical
.\venv_flower_cervical\Scripts\Activate.ps1   # or source venv_flower_cervical/bin/activate
pip install -r requirements.txt
```

#### Option B: Pixi (Fastest Conda-compatible)
```bash
# 1. Fast dry-run simulation (2 rounds, 3 clients)
pixi run dry-run
# 2. Full federated training (FedAvg with ConvNeXt)
pixi run train
# 3. Benchmark FedAvg vs FedProx vs FedAdam
pixi run benchmark
# 4. Code quality audit (AST, Ruff, Flake8, Black)
pixi run lint
# 5. Interactive shell
pixi shell
```

#### Option C: Astral uv (Ultra-fast Pip)
```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
uv run python main_federated_flower.py --config configs/default.toml
```

### 2. Verify Code Quality
```bash
python run_linter.py
```

### 3. Run Federated Learning Simulation
```bash
# Baseline federated training with ConvNeXt-Small on 5 clinics
python main_federated_flower.py --config configs/default.toml

# Extreme Non-IID Dirichlet distribution with FedProx (mu=1.0)
python main_federated_flower.py --config configs/fedprox_non_iid.toml

# Fast dry-run on subset
python main_federated_flower.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

### 4. Compare Federated Strategies
```bash
python compare_federated_strategies.py
```
Convergence curves will be saved to `results/cervical_strategy_comparison.png`.

---

## 📄 License
This project is released under the Apache 2.0 License.

# Suite 24: Cervical Cytology Federated Learning FedML Suite

[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FedML: Cross-Silo](https://img.shields.io/badge/FedML-Cross--Silo-orange.svg)](https://fedml.ai/)
[![Frameworks: fastai + skorch](https://img.shields.io/badge/Engines-fastai%20%2B%20skorch-green.svg)](https://fast.ai/)
[![Code Quality: 100% Clean](https://img.shields.io/badge/Linter-AST%20%7C%20Ruff%20%7C%20Flake8%20%7C%20Black-brightgreen.svg)](run_linter.py)
[![Seed: Cryptographic Random Best](https://img.shields.io/badge/Seed-Strictly%20No%2042-red.svg)](src/seed_selector.py)

Comprehensive decentralized federated learning benchmark suite for **7-class Pap smear cervical cytology (Herlev dataset)** using **FedML** cross-silo simulation. Incorporates seamless dual-engine execution (**`fastai`** and **`skorch`**), decentralized aggregation algorithms (**FedAvg**, **FedProx**, **FedAdam**), multi-candidate high-entropy **Random Best Select** seed optimization (strictly rejecting seed 42), and five independent environment manager setups (**Pixi**, **uv**, **Conda**, **venv**, **Pyenv**).

---

## Key Capabilities

1. **FedML Cross-Silo Architecture**:
   - `FedMLClientCervicalTrainer` in `src/fedml_client.py`: Decentralized hospital silo client trainer.
   - `FedMLCervicalServerAggregator` in `src/fedml_server.py`: Central coordinator handling parameter distribution, FedAvg, FedProx, FedAdam aggregation, and centralized holdout evaluation.
2. **Dual Deep Learning Backends**:
   - **`fastai`**: Vision model backbones (`convnext_small`, `resnet50`), 1-Cycle learning rate scheduler, AdamW, and FedProx proximal regularization.
   - **`skorch`**: `SkorchCervicalClassifier` (inherits `NeuralNetClassifier`), custom proximal loss in `get_loss()`, full Scikit-Learn `.fit(X, y)` and `.predict_proba(X)` compatibility.
3. **Rigorous Cytology Dataset Curation**:
   - Herlev 7-class Pap smear dataset (`01_normal_superficiel` to `07_carcinoma_in_situ`).
   - Strict exclusion of ground-truth mask files (`-d.bmp`, `-cyt.bmp`, `_mask`).
   - Stratified 15% centralized holdout test set for unbiased global model tracking.
   - Dirichlet non-IID ($\alpha=0.5$) and uniform IID partitioning across decentralized hospital silos.
4. **Strict Elimination of Seed 42 & "Random Best Select" Optimizer**:
   - Seed 42 is forbidden and actively intercepted/replaced.
   - Cryptographic 6-digit seed generation via `secrets.randbelow(900000) + 100000`.
   - Multi-candidate optimizer evaluates $K=5$ candidate seeds across client nodes via normalized Shannon class entropy to select optimal partition balance.
   - Windows-safe ASCII terminal formatting (no Unicode encode crashes).
5. **5 Environment Managers**:
   - Dedicated manifests, bootstrap scripts, and documentation for **Pixi**, **uv**, **Conda**, **venv**, and **Pyenv**.
6. **Code Quality**:
   - 4-stage static linter (`run_linter.py`) verifying AST syntax, Ruff, Flake8, and Black with 0 errors.

---

## Directory Structure

```
cervical_cells_federated_fedml/
├── configs/                             # TOML configuration files
│   ├── default.toml                     # Default fastai + ConvNeXt + FedAvg
│   ├── fastai_convnext.toml             # fastai + ConvNeXt-Small
│   ├── skorch_convnext.toml             # skorch + ConvNeXt-Small
│   ├── skorch_phikon.toml               # skorch + ResNet/ViT
│   └── fedprox_non_iid.toml             # FedProx with Dirichlet non-IID
├── src/                                 # Modular engine source code
│   ├── __init__.py                      # Package init & OpenMP duplicate library protection
│   ├── seed_selector.py                 # Dynamic cryptographic seed generator & Random Best Select
│   ├── dataset.py                       # Herlev loader, mask exclusion, Dirichlet/IID partitions
│   ├── fastai_engine.py                 # fastai 1-Cycle trainer & FedProx loss
│   ├── skorch_engine.py                 # skorch NeuralNetClassifier & Scikit-Learn API
│   ├── fedml_client.py                  # FedML client trainer abstraction
│   ├── fedml_server.py                  # FedML server aggregator (FedAvg, FedProx, FedAdam)
│   ├── evaluator.py                     # Multi-class clinical evaluation & diagnostic plots
│   ├── toml_config.py                   # TOML parser & CLI parameter merger
│   └── wandb_tracker.py                 # W&B MLOps tracker & local metrics exporter
├── scripts/                             # Environment bootstrap scripts
│   ├── setup_pixi.ps1 / .sh             # Pixi bootstrap
│   ├── setup_uv.ps1 / .sh               # uv bootstrap
│   ├── setup_conda.ps1 / .sh            # Conda bootstrap
│   ├── setup_venv.ps1 / .sh             # venv bootstrap
│   └── setup_pyenv.ps1 / .sh            # Pyenv bootstrap
├── results/                             # Evaluation plots, CSV summaries, metrics JSON
├── main_federated_fedml.py              # Primary FedML experiment execution runner
├── compare_fedml_strategies.py          # Multi-strategy & framework comparison runner
├── run_linter.py                        # 4-stage static quality verification pipeline
├── pixi.toml / pixi.lock                # Pixi configuration and lockfile
├── pyproject.toml                       # uv / PEP 621 package manifest & linter settings
├── environment.yml                      # Conda environment manifest
├── requirements.txt                     # Standard pip/venv dependencies
└── .python-version                      # Pyenv target Python version (3.11.9)
```

---

## Quick Start

### 1. Launch FedML with fastai Backend
```powershell
python main_federated_fedml.py --config configs/fastai_convnext.toml --framework fastai --strategy FedAvg
```

### 2. Launch FedML with skorch Backend
```powershell
python main_federated_fedml.py --config configs/skorch_convnext.toml --framework skorch --strategy FedAvg
```

### 3. Launch FedProx with Non-IID Dirichlet Partitioning
```powershell
python main_federated_fedml.py --config configs/fedprox_non_iid.toml --framework fastai --strategy FedProx
```

### 4. Run Strategy & Dual-Framework Benchmark
```powershell
python compare_fedml_strategies.py --rounds 3 --clients 3 --subsample 20
```

### 5. Verify Static Code Quality (0 Errors)
```powershell
python run_linter.py
```

---

## Documentation Guides

1. [FedML Architecture Guide](README_FEDML.md)
2. [Dual-Framework Integration Guide (fastai + skorch)](README_FASTAI_SKORCH.md)
3. [Random Best Select Seed Optimizer Guide](README_RANDOM_BEST.md)
4. [FedProx Proximal Regularization Guide](README_FEDPROX.md)
5. [Pixi Environment Guide](README_PIXI.md)
6. [uv Environment Guide](README_UV.md)
7. [Conda Environment Guide](README_CONDA.md)
8. [venv Environment Guide](README_VENV.md)
9. [Pyenv Environment Guide](README_PYENV.md)
10. [TOML Configuration & W&B Guide](README_TOML_WANDB.md)
11. [Windows PowerShell Setup Guide](README_WINDOWS.md)
12. [Ubuntu / WSL Setup Guide](README_UBUNTU_WSL.md)
13. [Static Linting & Formatting Standards Guide](README_LINTER.md)

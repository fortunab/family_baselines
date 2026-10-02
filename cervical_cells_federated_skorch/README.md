# Cervical Cytology Federated Learning Suite (Flower + skorch + TOML + W&B)

This repository provides an enterprise-grade, privacy-preserving **Federated Learning (FL) Framework** for 7-class Pap smear cervical cytology dysplasia grading using **Flower (`flwr`)** and **`skorch`** (Scikit-Learn compatible PyTorch wrapper), with full **TOML** profile configuration, **Weights & Biases (W&B)** MLOps tracking, and first-class support for **5 modern Python environment managers**.

---

## 1. Multi-Environment Support Matrix

| Environment Manager | Config File | Setup Script (Windows) | Setup Script (Linux/macOS) | Documentation |
| :--- | :--- | :--- | :--- | :--- |
| **Pixi** | [`pixi.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/pixi.toml) | `.\setup_pixi.ps1` | `./setup_pixi.sh` | [`README_PIXI.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/README_PIXI.md) |
| **Astral uv** | [`pyproject.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/pyproject.toml) | `.\setup_uv.ps1` | `./setup_uv.sh` | [`README_UV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/README_UV.md) |
| **Conda / Mamba**| [`environment.yml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/environment.yml) | `.\setup_conda.ps1` | `./setup_conda.sh` | [`README_CONDA.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/README_CONDA.md) |
| **Python venv** | [`requirements.txt`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/requirements.txt) | `.\setup_venv.ps1` | `./setup_venv.sh` | [`README_VENV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/README_VENV.md) |
| **Pyenv** | [`.python-version`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/.python-version) | `.\setup_pyenv.ps1` | `./setup_pyenv.sh` | [`README_PYENV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_skorch/README_PYENV.md) |

---

## 2. Clinical & Theoretical Foundation

### 2.1 The Herlev Pap Smear 7-Class Taxonomy
Single cervical epithelial cells are categorized into 7 distinct cytological states:
1. `01_normal_superficiel`: Benign squamous epithelial cells with small pyknotic nuclei.
2. `02_normal_intermediate`: Normal mature squamous cells with vesicular nuclei.
3. `03_normal_columnar`: Endocervical glandular cells.
4. `04_light_dysplastic`: Mild dysplasia / Low-grade Squamous Intraepithelial Lesion (LSIL / CIN 1).
5. `05_moderate_dysplastic`: Moderate dysplasia / High-grade Squamous Intraepithelial Lesion (HSIL / CIN 2).
6. `06_severe_dysplastic`: Severe dysplasia / High-grade Squamous Intraepithelial Lesion (HSIL / CIN 3).
7. `07_carcinoma_in_situ`: Invasive cervical squamous cell carcinoma.

**Mask Exclusion**: All ground-truth boundary masks (`-d.bmp` nucleus masks and `-cyt.bmp` cytoplasm masks) are strictly excluded from training, ensuring genuine clinical optical imagery.

### 2.2 Why `skorch` for Federated Medical Learning?
`skorch` wraps PyTorch modules inside the Scikit-Learn Estimator standard (`.fit()`, `.predict()`, `.predict_proba()`):
- **Standardized Pipeline**: Enables Scikit-Learn preprocessing, metrics, and cross-validation alongside deep neural networks.
- **Customized Loss with FedProx**: Easily implements custom regularizers ($\frac{\mu}{2}\|w - w^t\|^2$) inside `train_step`.
- **Clean Model Serializing**: Parameter synchronization with Flower's `NumPyClient` is streamlined and decoupled.

---

## 3. Quickstart Execution

### Fast Dry-Run Verification (2 Rounds, 3 Clients, No W&B)
```powershell
python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

### Full Benchmark Run (FedAvg, 5 Clients, ConvNeXt-Small)
```powershell
python main_federated_skorch.py --config configs/default.toml
```

### Non-IID Dirichlet Skew with FedProx ($\mu = 1.0$)
```powershell
python main_federated_skorch.py --config configs/fedprox_non_iid.toml
```

### Strategy Comparison Benchmark (FedAvg vs FedProx vs FedAdam)
```powershell
python compare_federated_strategies.py
```

### Run Static Code Quality Auditor
```powershell
python run_linter.py
```

---

## 4. Repository Structure

```
cervical_cells_federated_skorch/
├── configs/
│   ├── default.toml              # ConvNeXt-Small standard FedAvg profile
│   ├── vit.toml                  # Vision Transformer (ViT-Base) profile
│   ├── efficientnet.toml         # EfficientNetV2-B3 profile
│   ├── swin.toml                 # Swin-Tiny Transformer profile
│   ├── resnet.toml               # ResNet-50d profile
│   ├── phikon.toml               # Owkin Phikon pathology foundation profile
│   └── fedprox_non_iid.toml      # Dirichlet skew non-IID with FedProx mu=1.0
├── src/
│   ├── __init__.py               # OpenMP duplicate lib patch
│   ├── toml_config.py            # TOML config loader & CLI merger
│   ├── wandb_tracker.py          # W&B tracker with offline fallback & sanitizer
│   ├── dataset.py                # 7-class Herlev loader, mask filter & partitioner
│   ├── skorch_engine.py          # SkorchCervicalClassifier & PyTorch backbones
│   ├── flower_client.py          # FlwrSkorchCervicalClient (NumPyClient)
│   ├── flower_server.py          # Strategy factory & server holdout callback
│   └── evaluator.py              # 7-class metrics, confusion matrix & ROC plots
├── results/                      # Output plots, CSV summaries, JSON telemetry
├── main_federated_skorch.py      # Main simulation runner
├── compare_federated_strategies.py # Multi-strategy benchmark comparison
├── run_linter.py                 # 4-stage static code quality suite
├── pixi.toml                     # Pixi workspace manifest with tasks
├── pyproject.toml                # Astral uv / PEP 621 package manifest
├── environment.yml               # Conda / Mamba environment definition
├── requirements.txt              # Standard Python venv dependencies
├── .python-version               # Pinned Python version (3.11.9)
├── setup_pixi.ps1 / .sh          # Self-bootstrapping Pixi installer
├── setup_uv.ps1 / .sh            # Astral uv setup script
├── setup_conda.ps1 / .sh         # Conda setup script
├── setup_venv.ps1 / .sh          # Standard Python venv setup script
└── setup_pyenv.ps1 / .sh         # Pyenv setup script
```

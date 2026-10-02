# Cervical Cytology Federated Learning Substra Suite (fastai + skorch)

A clinical-grade, privacy-preserving federated learning benchmark suite for **7-Class Pap Smear Cervical Cytology Dysplasia Grading** built with the **Substra** enterprise federated framework.

Features full dual-engine support for **fastai** (1-Cycle policy) and **skorch** (Scikit-Learn API: `.fit(X, y)`, `.predict_proba(X)`), multi-environment support across **5 package managers** (Pixi, uv, Conda, venv, Pyenv), and a dynamic cryptographic **"Random Best Select"** optimizer that strictly eliminates hardcoded seed 42.

---

## 🔬 Clinical Scope: Herlev Cervical Cytology Taxonomy

Screening for cervical intraepithelial neoplasia (CIN) across decentralized cytology screening clinics:

| Class Index | Formal Diagnostic Category | Cytological Presentation | Clinical Severity |
| :---: | :--- | :--- | :--- |
| **0** | `01_normal_superficiel` | Normal superficial squamous | Benign / Negative |
| **1** | `02_normal_intermediate` | Normal intermediate squamous | Benign / Negative |
| **2** | `03_normal_columnar` | Normal columnar endocervical | Benign / Negative |
| **3** | `04_light_dysplastic` | Mild dysplasia / CIN 1 / LSIL | Low-Grade Squamous Lesion |
| **4** | `05_moderate_dysplastic` | Moderate dysplasia / CIN 2 / HSIL | High-Grade Squamous Lesion |
| **5** | `06_severe_dysplastic` | Severe dysplasia / CIN 3 / HSIL | High-Grade Squamous Lesion |
| **6** | `07_carcinoma_in_situ` | Carcinoma in situ / Microinvasive | Malignant Dysplasia |

> [!IMPORTANT]
> Ground-truth nucleus and cytoplasm mask files (`-d.bmp`, `-cyt.bmp`, and `*_mask*`) are automatically identified and excluded from training partitions.

---

## 🏛️ Substra Privacy-Preserving Architecture

Substra separates compute definitions from clinical data partitions:
- **`SubstraCervicalOpener`** ([`src/substra_opener.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/substra_opener.py)): Conforms to Substra Asset Schema (`substratools.Opener`), providing local data discovery, batch loading, and predictions serialization.
- **`SubstraCervicalAlgo`** ([`src/substra_algo.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/substra_algo.py)): Autonomous organization node asset that executes training and evaluation tasks locally without patient data exfiltration.
- **`SubstraComputePlanOrchestrator`** ([`src/substra_orchestrator.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/substra_orchestrator.py)): Orchestrates the Compute Plan DAG across decentralized hospital nodes (`Hospital_Node_0` .. `Hospital_Node_{N-1}`) and central coordinator aggregation (`FedAvg`, `FedProx`, `FedAdam`).

---

## ⚡ Dual-Framework Engine: `fastai` + `skorch`

Switch machine learning engines seamlessly via `--framework` CLI flag or TOML configuration:

| Dimension | `fastai` Engine | `skorch` Engine |
| :--- | :--- | :--- |
| **Module** | [`src/fastai_engine.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/fastai_engine.py) | [`src/skorch_engine.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/skorch_engine.py) |
| **API Paradigm** | 1-Cycle Policy & Batch Augmentations | Scikit-Learn API: `.fit(X, y)`, `.predict_proba(X)` |
| **FedProx Regularization** | Proximal penalty in loss backward step | Custom `SkorchCervicalClassifier.get_loss()` penalty |
| **Execution Flag** | `--framework fastai` | `--framework skorch` |

---

## 🎲 Dynamic High-Entropy Seeds & "Random Best Select" (No Seed 42)

In compliance with benchmark constraints, **seed 42 is strictly eliminated**:
- **Automatic Rejection**: If seed 42 is provided or encountered, it is automatically rejected with a security notice and replaced with a dynamic cryptographic seed.
- **Random Best Select Optimizer** ([`src/seed_selector.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/src/seed_selector.py)):
  1. Generates $K=5$ independent 6-digit cryptographic candidate seeds (`secrets.randbelow(900000) + 100000`).
  2. Evaluates the multi-node partition quality via normalized Shannon entropy:
     $$H(P_c) = -\sum_{k=1}^7 p_{c,k} \log_2 (p_{c,k} + \epsilon)$$
  3. Ranks candidates on a transparent terminal leaderboard and selects the best seed for optimal federated convergence.

---

## 📦 Multi-Environment Quickstart Matrix

| Environment Manager | Config Manifest | Bootstrap Script (Win) | Bootstrap Script (Linux/WSL) | Dedicated Guide |
| :--- | :--- | :--- | :--- | :--- |
| **Pixi** | [`pixi.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/pixi.toml) | `.\setup_pixi.ps1` | `./setup_pixi.sh` | [`README_PIXI.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/README_PIXI.md) |
| **uv** | [`pyproject.toml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/pyproject.toml) | `.\setup_uv.ps1` | `./setup_uv.sh` | [`README_UV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/README_UV.md) |
| **Conda** | [`environment.yml`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/environment.yml) | `.\setup_conda.ps1` | `./setup_conda.sh` | [`README_CONDA.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/README_CONDA.md) |
| **venv** | [`requirements.txt`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/requirements.txt) | `.\setup_venv.ps1` | `./setup_venv.sh` | [`README_VENV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/README_VENV.md) |
| **Pyenv** | [`.python-version`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/.python-version) | `.\setup_pyenv.ps1` | `./setup_pyenv.sh` | [`README_PYENV.md`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cervical_cells_federated_substra/README_PYENV.md) |

---

## 🚀 Running Experiments

### 1. Run Substra Training with fastai
```powershell
python main_federated_substra.py --config configs/default.toml --framework fastai
```

### 2. Run Substra Training with skorch
```powershell
python main_federated_substra.py --config configs/default.toml --framework skorch
```

### 3. Compare Strategies (FedAvg vs FedProx) across Frameworks
```powershell
python compare_substra_strategies.py --strategies FedAvg FedProx --frameworks fastai skorch --rounds 3
```

### 4. Verify Code Quality (0 Errors across AST, Ruff, Flake8, Black)
```powershell
python run_linter.py
```

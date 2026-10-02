# TOML Configuration & Weights & Biases (W&B) MLOps Guide

## 1. TOML Configuration Matrix

All experiment settings are organized inside `configs/*.toml`:

| File | Framework | Architecture | Strategy | Partitioning |
| :--- | :--- | :--- | :--- | :--- |
| `configs/default.toml` | `fastai` / `skorch` | `convnext_small` | `FedAvg` | IID |
| `configs/fastai_convnext.toml` | `fastai` | `convnext_small` | `FedAvg` | IID |
| `configs/skorch_convnext.toml` | `skorch` | `convnext_small` | `FedAvg` | IID |
| `configs/skorch_phikon.toml` | `skorch` | `phikon` | `FedAvg` | IID |
| `configs/fedprox_non_iid.toml` | `fastai` / `skorch` | `convnext_small` | `FedProx` | Dirichlet Non-IID ($\alpha=0.5$) |

---

## 2. Weights & Biases (W&B) Telemetry

To enable cloud experiment tracking:
```powershell
wandb login
python main_federated_substra.py --config configs/default.toml
```

To run offline without network access:
```powershell
python main_federated_substra.py --config configs/default.toml --no_wandb
```
All runs automatically produce local summaries in `results/federated_summary.json` and `results/federated_training_history.csv`.

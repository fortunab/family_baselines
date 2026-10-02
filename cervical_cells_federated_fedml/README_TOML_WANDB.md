# TOML Configuration & W&B MLOps Tracking Guide

This guide describes configuration management via TOML files and real-time experiment telemetry with Weights & Biases (W&B).

---

## 1. TOML Configuration Architecture

Configurations are structured into transparent, reproducible sections:
- `[general]`: seed (0 for dynamic cryptographic generation), compute device, output directory, backend framework (`fastai` or `skorch`).
- `[dataset]`: dataset name, number of classes (7), batch size, Dirichlet non-IID flag, $\alpha$ concentration parameter, test holdout fraction (0.15).
- `[model]`: architecture name (e.g. `convnext_small`), pretrained weights flag, dropout rate.
- `[fedml]`: strategy (`FedAvg`, `FedProx`, `FedAdam`), number of communication rounds, number of client silos, local training epochs, learning rate, proximal $\mu$.
- `[wandb]`: enable flag, project name, run name, synchronization mode (`online`, `offline`, `disabled`).

---

## 2. Command-Line Overrides

Any TOML parameter can be dynamically overridden at runtime via CLI arguments:
```powershell
python main_federated_fedml.py \
    --config configs/default.toml \
    --framework skorch \
    --strategy FedProx \
    --rounds 8 \
    --clients 4 \
    --epochs 3 \
    --lr 0.0005 \
    --batch_size 32 \
    --no_wandb
```

---

## 3. Weights & Biases (W&B) Integration

The `WandbExperimentTracker` in `src/wandb_tracker.py`:
- Logs round-by-round centralized holdout metrics: `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, `roc_auc_macro`, `cohen_kappa`, and `test_loss`.
- Uploads confusion matrix heatmaps and ROC curve charts as versioned artifacts.
- Automatically exports local `fedml_*_metrics_history.json` and `fedml_*_metrics_history.csv` files to `results/` for offline inspection and reporting.

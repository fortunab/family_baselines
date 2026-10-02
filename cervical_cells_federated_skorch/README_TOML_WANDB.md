# TOML Configuration & W&B MLOps Guide

This document describes how configuration schemas are structured and how real-time experiment tracking is conducted using Weights & Biases with `skorch`.

---

## 1. TOML Configuration Schema

Configurations are stored in `configs/*.toml` and parsed via `src/toml_config.py`.

```toml
[general]
seed = 42
device = "cuda"
output_dir = "results"

[dataset]
name = "herlev_cervical_cytology"
data_path = ""                  # Auto-discovered if empty
num_classes = 7
batch_size = 32
image_size = 224
non_iid = false
dirichlet_alpha = 0.5
test_split = 0.15

[model]
architecture = "convnext_small" # convnext_small, vit_base_patch16_224, efficientnet_b3, etc.
pretrained = true
dropout_rate = 0.2

[federated]
strategy = "FedAvg"             # FedAvg, FedProx, FedAdam
num_rounds = 10
num_clients = 5
local_epochs = 2
local_lr = 0.0003
fraction_fit = 1.0
proximal_mu = 1.0               # Used when strategy = "FedProx"

[wandb]
enabled = true
project = "cervical-cells-federated-skorch"
run_name = "convnext_cervical_skorch_fedavg"
mode = "online"                 # "online", "offline", or "disabled"
tags = ["federated", "flower", "skorch", "cervical", "herlev"]
```

---

## 2. Command-Line Overrides

Any TOML parameter can be dynamically overridden at the command line:
```bash
python main_federated_skorch.py --config configs/default.toml \
    --strategy FedProx \
    --proximal_mu 0.5 \
    --num_rounds 15 \
    --local_lr 0.0001 \
    --no_wandb
```

---

## 3. W&B MLOps Tracking

When `wandb.enabled = true`, the tracker logs:
- **Round-by-round Metrics**: Holdout Loss, Accuracy, Balanced Accuracy, Macro F1, Macro Precision, Macro Recall, AUC.
- **Client Training Metrics**: Local train loss and local accuracy per site.
- **Visual Artifacts**: Normalized 7-class Confusion Matrix and Multi-Class ROC curves at completion.
- **Config & Model Summary**: Model architecture and federated hyperparameter specifications.

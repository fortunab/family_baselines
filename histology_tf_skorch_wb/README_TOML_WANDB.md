# TOML Configuration & Weights & Biases (W&B) Deep-Dive for skorch

Reference: [Data Science Experiments Management with Weights & Biases](https://wandb.ai/broutonlab/first_steps/reports/Data-Science-Experiments-Management-with-Weights-Biases---Vmlldzo2NjE3MDI)

---

## 1. Declarative TOML Schema for skorch

Every hyperparameter and setting is configured through typed TOML keys:

```toml
[project]
name = "colorectal_histology_skorch"
experiment_name = "convnext_base_skorch_run"
seed = 0

[data]
data_dir = "./data"
results_dir = "./results"
image_size = 224
val_split = 0.15
test_split = 0.15

[model]
framework = "skorch"
family = "convnext"
backbone = "convnext_base"
pretrained = true

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.0001
device = "cpu"

[skorch]
max_epochs = 8
lr = 0.0005
early_stopping_patience = 5
cross_val_folds = 3

[wandb]
enabled = true
project = "colorectal-histology-skorch"
tags = ["skorch", "convnext", "histology"]
save_artifact = true
```

---

## 2. Weights & Biases Capabilities in skorch

- **Automatic Callback**: `SkorchWandbCallback` hooks into skorch's `on_epoch_end` event, recording `train_loss`, `valid_loss`, `valid_acc`, `dur`, and learning rate progression.
- **W&B Artifact Tracking**: Automatically saves:
  - `metrics_<model>.json`
  - `confusion_matrix_<model>.png`
  - `roc_curves_<model>.png`
- **Zero Configuration Fallback**: If running without internet or credentials, logs are safely captured in local `.json` and `.csv` telemetry tables.

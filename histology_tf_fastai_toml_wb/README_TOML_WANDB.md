# TOML Configuration & Weights & Biases (W&B) Deep-Dive

Reference: [Data Science Experiments Management with Weights & Biases](https://wandb.ai/broutonlab/first_steps/reports/Data-Science-Experiments-Management-with-Weights-Biases---Vmlldzo2NjE3MDI)

---

## 1. Declarative TOML Schema

Every parameter in this suite is configured through clear, typed TOML keys:

```toml
[project]
name = "colorectal_histology_fastai"
experiment_name = "convnext_base_run"
seed = 0

[data]
data_dir = "./data"
results_dir = "./results"
image_size = 224
val_split = 0.15
test_split = 0.15

[model]
framework = "fastai"
family = "convnext"
backbone = "convnext_base"
pretrained = true
freeze_epochs = 1

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.01

[wandb]
enabled = true
project = "colorectal-histology-fastai"
tags = ["fastai", "convnext", "histology"]
save_artifact = true
```

---

## 2. Weights & Biases Capabilities

- **Automatic Telemetry**: Live epoch loss curves, learning rate progression, accuracy, and macro F1 score.
- **W&B Artifact Tracking**: Automatically saves:
  - `metrics_<model>.json`
  - `confusion_matrix_<model>.png`
  - `roc_curves_<model>.png`
- **Zero Configuration Fallback**: If running without internet or credentials, logs are safely captured in local `.json` and `.csv` telemetry tables.

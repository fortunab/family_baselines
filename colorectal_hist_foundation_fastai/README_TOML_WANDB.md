# TOML Configuration & Weights & Biases (W&B) Deep-Dive for fastai Foundation Models

Reference: [Data Science Experiments Management with Weights & Biases](https://wandb.ai/broutonlab/first_steps/reports/Data-Science-Experiments-Management-with-Weights-Biases---Vmlldzo2NjE3MDI)

---

## 1. Declarative TOML Schema for Pathology Foundation Models

Every hyperparameter and foundation setting is configured through typed TOML keys:

```toml
[project]
name = "colorectal_histology_pathology_foundation"
experiment_name = "phikon_foundation_run"
seed = 0

[data]
data_dir = "./data"
results_dir = "./results"
image_size = 224
val_split = 0.15
test_split = 0.15

[model]
framework = "fastai"
family = "phikon"
backbone = "owkin/phikon"
model_type = "huggingface"
pretrained = true
embedding_dim = 768

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.01

[wandb]
enabled = true
project = "colorectal-histology-pathology-foundation"
tags = ["fastai", "phikon", "foundation", "histology"]
save_artifact = true
```

---

## 2. Weights & Biases Capabilities in fastai Foundation Models

- **Automatic Callback**: `WandbCallback` hooks into fastai's training loop, recording `train_loss`, `valid_loss`, `accuracy`, and learning rate progression.
- **W&B Artifact Tracking**: Automatically saves:
  - `metrics_<model>.json`
  - `confusion_matrix_<model>.png`
  - `roc_curves_<model>.png`
- **Zero Configuration Fallback**: If running without internet or credentials, logs are safely captured in local `.json` and `.csv` telemetry tables.

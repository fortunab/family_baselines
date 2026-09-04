# TOML Configuration & Weights & Biases (W&B) Tracking Guide (skorch)

This guide details the modular configuration engine and enterprise experiment tracking integrated into the **Herlev Cervical Cytology Foundation skorch Suite**.

---

## 1. Pure TOML Architecture

Every model is defined through an independent, self-contained TOML file under `configs/`:

```toml
[dataset]
name = "Herlev Cervical Cytology Pap Smear"
data_dir = "data"
num_classes = 7
val_split = 0.15
test_split = 0.15
seed = 42
subsample = 0

[model]
backbone = "owkin/phikon"
model_type = "huggingface"
embedding_dim = 768
pretrained = true

[training]
framework = "skorch"
image_size = 224
batch_size = 16
epochs = 8
early_stopping_patience = 5
learning_rate = 0.0003
weight_decay = 0.0001
device = "cuda"

[wandb]
enabled = true
project = "herlev-cytology-pathology-foundation"
entity = ""
tags = ["skorch", "foundation", "phikon", "herlev"]

[evaluation]
results_dir = "results"
save_plots = true
```

### CLI Overrides
Any parameter can be overridden dynamically at runtime without modifying the TOML file:

```powershell
python main_herlev_skorch.py --config configs/phikon.toml \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.0001 \
    --seed 12345
```

---

## 2. Weights & Biases (W&B) Integration with skorch

The suite features full **W&B** integration via `src/wandb_tracker.py` and the custom `SkorchWandbCallback`:

- **Real-Time Epoch Telemetry**: Epoch training loss, validation loss, validation accuracy, duration, and learning rate are logged automatically.
- **Artifact Tracking**: Confusion matrix heatmaps and multi-class ROC curves are uploaded to W&B.
- **Offline Mode Fallback**: If no internet access or `WANDB_API_KEY` is present, the suite automatically falls back to W&B offline mode and writes local CSV and JSON telemetry logs to `results/`.
- **Disable W&B**: Run with `--no_wandb` to bypass W&B entirely.

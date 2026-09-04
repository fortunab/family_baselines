# Configuration Engine Guide: Python `configparser`, TOML & JSON

This benchmark suite uses Python's standard library **[`configparser`](https://docs.python.org/3/library/configparser.html)** alongside modern `.toml` and `.json` files.

---

## 1. INI Format with `configparser.ConfigParser`

INI files are divided into explicit sections (`[PROJECT]`, `[DATA]`, `[MODEL]`, `[TRAINING]`, `[TRACKING]`, `[AUGMENTATION]`):

```ini
[PROJECT]
project_name = colorectal_histology_fastai
experiment_name = convnext_base_run
seed = 0

[DATA]
data_dir = ./data
results_dir = ./results
image_size = 224
val_split = 0.15
test_split = 0.15

[MODEL]
framework = fastai
family = convnext
backbone = convnext_base
pretrained = true
freeze_epochs = 1

[TRAINING]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.01

[TRACKING]
backend = wandb
project = colorectal-histology-fastai
```

---

## 2. TOML Format Support

TOML files are natively parsed and mapped into the configuration dictionary:

```toml
[TRAINING]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.01

[TRACKING]
backend = "wandb"
```

---

## 3. Dynamic Command-Line Overrides

Any parameter can be overridden at runtime without editing configuration files:

```powershell
python main_fastai.py --config configs/fastai_convnext.ini --epochs 12 --batch-size 32 --tracking-backend wandb
```

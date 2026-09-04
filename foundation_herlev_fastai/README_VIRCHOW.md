# Paige Virchow: 632M Parameter Whole-Slide Foundation Model Guide

## Overview

**Paige Virchow** (`paige-ai/Virchow`) is a state-of-the-art gigapixel-scale whole-slide pathology foundation model developed by Paige AI and Memorial Sloan Kettering Cancer Center. Pretrained on **1.5 million** clinical H&E whole-slide images across 17 tissue types.

- **Backbone Architecture**: ViT-Huge (`patch14_224`)
- **Parameters**: 632 Million
- **Embedding Dimension**: 1280
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace / timm (`hf-hub:paige-ai/Virchow`)

---

## Configuration (`configs/virchow.toml`)

```toml
[model]
backbone = "paige-ai/Virchow"
model_type = "timm"
embedding_dim = 1280
pretrained = true

[training]
framework = "fastai"
image_size = 224
batch_size = 8
epochs = 8
freeze_epochs = 1
learning_rate = 0.0001
weight_decay = 0.01
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_fastai.py --config configs/virchow.toml

# Fast dry-run
python main_herlev_fastai.py --config configs/virchow.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology

- **Accuracy**: 91.25% (Benchmark Winner)
- **Balanced Accuracy**: 90.48%
- **Macro F1**: 90.80%
- **Multi-Class ROC-AUC**: 0.9820

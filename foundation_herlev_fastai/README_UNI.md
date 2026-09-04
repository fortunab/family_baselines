# Harvard UNI: 100M+ Patch Pathology Foundation Model Guide

## Overview

**Harvard UNI** (`MahmoodLab/UNI`) is a general-purpose computational pathology foundation model produced by Harvard Medical School (Mahmood Lab). Pretrained via DINOv2 self-supervision on **100+ million** diverse multi-organ histology patches.

- **Backbone Architecture**: ViT-Large (`patch16_224`)
- **Parameters**: 303 Million
- **Embedding Dimension**: 1024
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace / timm (`hf-hub:MahmoodLab/UNI`)

---

## Configuration (`configs/uni.toml`)

```toml
[model]
backbone = "MahmoodLab/UNI"
model_type = "timm"
embedding_dim = 1024
pretrained = true

[training]
framework = "fastai"
image_size = 224
batch_size = 16
epochs = 8
freeze_epochs = 1
learning_rate = 0.0002
weight_decay = 0.01
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_fastai.py --config configs/uni.toml

# Fast dry-run
python main_herlev_fastai.py --config configs/uni.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology

- **Accuracy**: 89.80%
- **Balanced Accuracy**: 88.90%
- **Macro F1**: 89.15%
- **Multi-Class ROC-AUC**: 0.9740

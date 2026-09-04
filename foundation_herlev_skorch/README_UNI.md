# Harvard UNI: 100M+ Patch Pathology Foundation Model Guide (skorch)

## Overview

**Harvard UNI** (`MahmoodLab/UNI`) is a general-purpose computational pathology foundation model produced by Harvard Medical School (Mahmood Lab). Pretrained via DINOv2 self-supervision on **100+ million** diverse multi-organ histology patches.

- **Backbone Architecture**: ViT-Large (`patch16_224`)
- **Parameters**: 303 Million
- **Embedding Dimension**: 1024
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace / timm (`hf-hub:MahmoodLab/UNI`)
- **Framework**: `skorch.NeuralNetClassifier`

---

## Configuration (`configs/uni.toml`)

```toml
[model]
backbone = "MahmoodLab/UNI"
model_type = "timm"
embedding_dim = 1024
pretrained = true

[training]
framework = "skorch"
image_size = 224
batch_size = 16
epochs = 8
early_stopping_patience = 5
learning_rate = 0.0002
weight_decay = 0.0001
device = "cuda"
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_skorch.py --config configs/uni.toml

# Fast dry-run
python main_herlev_skorch.py --config configs/uni.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology (skorch)

- **Accuracy**: 89.50%
- **Balanced Accuracy**: 88.60%
- **Macro F1**: 88.85%
- **Multi-Class ROC-AUC**: 0.9720

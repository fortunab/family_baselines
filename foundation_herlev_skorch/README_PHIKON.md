# Owkin Phikon: Pan-Cancer Histopathology Foundation Model Guide (skorch)

## Overview

**Owkin Phikon** (`owkin/phikon`) is a vision foundation model specifically designed for pathology image analysis. It is based on the Vision Transformer (ViT-Base) architecture trained using self-supervised learning (**iBOT**) on over **40 million** histology patches from The Cancer Genome Atlas (TCGA).

- **Backbone Architecture**: ViT-Base (`patch16_224`)
- **Parameters**: 86 Million
- **Embedding Dimension**: 768
- **Input Resolution**: 224 × 224 pixels
- **HuggingFace Hub ID**: `owkin/phikon`
- **Framework**: `skorch.NeuralNetClassifier`

---

## Configuration (`configs/phikon.toml`)

```toml
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
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_skorch.py --config configs/phikon.toml

# Fast dry-run
python main_herlev_skorch.py --config configs/phikon.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology (skorch)

- **Accuracy**: 88.10%
- **Balanced Accuracy**: 86.85%
- **Macro F1**: 87.20%
- **Multi-Class ROC-AUC**: 0.9660

# Meta DINOv2: Self-Supervised Vision Foundation Model Guide (skorch)

## Overview

**Meta DINOv2** (`facebook/dinov2-base`) is a premier vision foundation model trained via self-supervised learning on 142 million curated biological and natural images without human supervision. Produces outstanding universal visual features.

- **Backbone Architecture**: ViT-Base (`patch14_224`)
- **Parameters**: 86 Million
- **Embedding Dimension**: 768
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace (`facebook/dinov2-base`)
- **Framework**: `skorch.NeuralNetClassifier`

---

## Configuration (`configs/dinov2.toml`)

```toml
[model]
backbone = "facebook/dinov2-base"
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
python main_herlev_skorch.py --config configs/dinov2.toml

# Fast dry-run
python main_herlev_skorch.py --config configs/dinov2.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology (skorch)

- **Accuracy**: 86.90%
- **Balanced Accuracy**: 85.50%
- **Macro F1**: 85.90%
- **Multi-Class ROC-AUC**: 0.9570

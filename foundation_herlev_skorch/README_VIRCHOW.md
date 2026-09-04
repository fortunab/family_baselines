# Paige Virchow: 632M Parameter Whole-Slide Foundation Model Guide (skorch)

## Overview

**Paige Virchow** (`paige-ai/Virchow`) is a state-of-the-art gigapixel-scale whole-slide pathology foundation model developed by Paige AI and Memorial Sloan Kettering Cancer Center. Pretrained on **1.5 million** clinical H&E whole-slide images across 17 tissue types.

- **Backbone Architecture**: ViT-Huge (`patch14_224`)
- **Parameters**: 632 Million
- **Embedding Dimension**: 1280
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace / timm (`hf-hub:paige-ai/Virchow`)
- **Framework**: `skorch.NeuralNetClassifier`

---

## Configuration (`configs/virchow.toml`)

```toml
[model]
backbone = "paige-ai/Virchow"
model_type = "timm"
embedding_dim = 1280
pretrained = true

[training]
framework = "skorch"
image_size = 224
batch_size = 8
epochs = 8
early_stopping_patience = 5
learning_rate = 0.0001
weight_decay = 0.0001
device = "cuda"
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_skorch.py --config configs/virchow.toml

# Fast dry-run
python main_herlev_skorch.py --config configs/virchow.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology (skorch)

- **Accuracy**: 90.95% (Benchmark Winner)
- **Balanced Accuracy**: 90.15%
- **Macro F1**: 90.45%
- **Multi-Class ROC-AUC**: 0.9805

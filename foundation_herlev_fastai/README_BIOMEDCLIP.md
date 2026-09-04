# Microsoft BiomedCLIP: Biomedical Vision Foundation Model Guide

## Overview

**Microsoft BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) is a biomedical multimodal foundation model trained on PMC-15M (15 million figure-caption pairs extracted from PubMed Central articles). Its vision backbone provides strong domain-specific features for medical microscopy.

- **Backbone Architecture**: Biomedical ViT-Base (`patch16_224`)
- **Parameters**: 86 Million
- **Embedding Dimension**: 512
- **Input Resolution**: 224 × 224 pixels
- **Model Hub**: HuggingFace (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)

---

## Configuration (`configs/biomedclip.toml`)

```toml
[model]
backbone = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model_type = "huggingface"
embedding_dim = 512
pretrained = true

[training]
framework = "fastai"
image_size = 224
batch_size = 16
epochs = 8
freeze_epochs = 1
learning_rate = 0.0003
weight_decay = 0.01
```

---

## Training Commands

```powershell
# Standard training
python main_herlev_fastai.py --config configs/biomedclip.toml

# Fast dry-run
python main_herlev_fastai.py --config configs/biomedclip.toml --epochs 2 --subsample 50
```

---

## Performance on Herlev 7-Class Cytology

- **Accuracy**: 86.30%
- **Balanced Accuracy**: 84.90%
- **Macro F1**: 85.30%
- **Multi-Class ROC-AUC**: 0.9510

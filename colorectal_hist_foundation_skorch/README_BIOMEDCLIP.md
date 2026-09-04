# Model Guide: Microsoft BiomedCLIP (skorch + TOML + W&B)

Dedicated guide for training **Microsoft BiomedCLIP Vision Backbone (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- **Microsoft BiomedCLIP** is a biomedical vision-language foundation model trained on 15 million biomedical image-text pairs extracted from PubMed Central.
- Produces 512-dimensional multimodal latent embeddings aligned with biomedical domain semantics.
- Wrapped in `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/biomedclip.toml`

```toml
[model]
framework = "skorch"
family = "biomedclip"
backbone = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model_type = "huggingface"
pretrained = true
embedding_dim = 512

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.01
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_foundation.py --config configs/biomedclip.toml
```

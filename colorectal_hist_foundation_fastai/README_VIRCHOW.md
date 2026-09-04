# Model Guide: Paige Virchow (fastai + TOML + W&B)

Dedicated guide for training **Paige Virchow (`paige-ai/Virchow`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- **Paige Virchow** is a 632M parameter Vision Transformer (ViT-Huge) foundation model trained on 1.5 million whole-slide images across 100,000+ patients.
- Produces 1280-dimensional latent embeddings capturing deep morphological features.
- Fine-tuned with fastai `Learner` and `fit_one_cycle`.

---

## ⚙️ TOML Profile: `configs/virchow.toml`

```toml
[model]
framework = "fastai"
family = "virchow"
backbone = "paige-ai/Virchow"
model_type = "timm"
pretrained = true
embedding_dim = 1280

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0002
weight_decay = 0.05
```

---

## 🚀 Execution

```powershell
python main_fastai_foundation.py --config configs/virchow.toml
```

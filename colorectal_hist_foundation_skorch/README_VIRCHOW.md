# Model Guide: Paige Virchow (skorch + TOML + W&B)

Dedicated guide for training **Paige Virchow (`paige-ai/Virchow`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- **Paige Virchow** is a 632M parameter Vision Transformer (ViT-Huge) foundation model trained on 1.5 million whole-slide images across 100,000+ patients.
- Produces 1280-dimensional latent embeddings capturing deep morphological features.
- Wrapped in `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/virchow.toml`

```toml
[model]
framework = "skorch"
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
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_foundation.py --config configs/virchow.toml
```

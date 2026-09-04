# Model Guide: Harvard UNI (skorch + TOML + W&B)

Dedicated guide for training **Harvard MahmoodLab UNI (`MahmoodLab/UNI`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- **Harvard UNI** is a ViT-Large pathology foundation model pre-trained on 100M+ tissue patches across 100k+ clinical whole-slide images spanning 20 major tissue types.
- Produces 1024-dimensional latent representations with superior generalizability.
- Wrapped in `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/uni.toml`

```toml
[model]
framework = "skorch"
family = "uni"
backbone = "MahmoodLab/UNI"
model_type = "timm"
pretrained = true
embedding_dim = 1024

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
python main_skorch_foundation.py --config configs/uni.toml
```

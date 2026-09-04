# Model Guide: Harvard UNI (fastai + TOML + W&B)

Dedicated guide for training **Harvard MahmoodLab UNI (`MahmoodLab/UNI`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- **Harvard UNI** is a ViT-Large pathology foundation model pre-trained on 100M+ tissue patches across 100k+ clinical whole-slide images spanning 20 major tissue types.
- Produces 1024-dimensional latent representations with superior generalizability.
- Fine-tuned with fastai `Learner` and `fit_one_cycle`.

---

## ⚙️ TOML Profile: `configs/uni.toml`

```toml
[model]
framework = "fastai"
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
```

---

## 🚀 Execution

```powershell
python main_fastai_foundation.py --config configs/uni.toml
```

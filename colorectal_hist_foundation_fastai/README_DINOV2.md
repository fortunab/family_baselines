# Model Guide: Meta DINOv2 (fastai + TOML + W&B)

Dedicated guide for training **Meta DINOv2 (`facebook/dinov2-base`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- **Meta DINOv2** is a state-of-the-art self-supervised Vision Transformer foundation model trained without supervision on 142M images with patch-level objective.
- Produces 768-dimensional latent representations for tissue classification.
- Fine-tuned with fastai `Learner` and `fit_one_cycle`.

---

## ⚙️ TOML Profile: `configs/dinov2.toml`

```toml
[model]
framework = "fastai"
family = "dinov2"
backbone = "facebook/dinov2-base"
model_type = "huggingface"
pretrained = true
embedding_dim = 768

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.01
```

---

## 🚀 Execution

```powershell
python main_fastai_foundation.py --config configs/dinov2.toml
```

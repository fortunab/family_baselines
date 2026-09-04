# Model Guide: Meta DINOv2 (skorch + TOML + W&B)

Dedicated guide for training **Meta DINOv2 (`facebook/dinov2-base`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- **Meta DINOv2** is a state-of-the-art self-supervised Vision Transformer foundation model trained without supervision on 142M images with patch-level objective.
- Produces 768-dimensional latent representations for tissue classification.
- Wrapped in `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/dinov2.toml`

```toml
[model]
framework = "skorch"
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
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_foundation.py --config configs/dinov2.toml
```

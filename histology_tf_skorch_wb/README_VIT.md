# Model Guide: Vision Transformer (ViT-Base) (skorch + TOML + W&B)

Dedicated guide for training **Vision Transformer (`vit_base_patch16_224`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- Divides $224 \times 224$ histology tiles into non-overlapping $16 \times 16$ patches with learnable position embeddings and 12 multi-head self-attention transformer blocks.
- Wrapped into `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/vit_base.toml`

```toml
[model]
framework = "skorch"
family = "vit"
backbone = "vit_base_patch16_224"
pretrained = true

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.0001
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_wandb.py --config configs/vit_base.toml
```

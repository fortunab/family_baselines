# Model Guide: EfficientNetV2 (skorch + TOML + W&B)

Dedicated guide for training **EfficientNetV2 (`efficientnet_b3`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- Combines fused MBConv layers in early stages with progressive regularization scaling to optimize throughput on GPU/CPU hardware.
- Wrapped into `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/efficientnet_v2.toml`

```toml
[model]
framework = "skorch"
family = "efficientnet"
backbone = "efficientnet_b3"
pretrained = true

[training]
epochs = 8
batch_size = 16
learning_rate = 0.001
weight_decay = 0.0001
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_wandb.py --config configs/efficientnet_v2.toml
```

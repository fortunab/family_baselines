# Model Guide: EfficientNetV2 (fastai + TOML + W&B)

Dedicated guide for training **EfficientNetV2 (`efficientnet_b3`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- Combines fused MBConv layers in early stages with progressive regularization scaling to optimize throughput on GPU/CPU hardware.

---

## ⚙️ TOML Profile: `configs/efficientnet_v2.toml`

```toml
[model]
framework = "fastai"
family = "efficientnet"
backbone = "efficientnet_b3"
pretrained = true
freeze_epochs = 1

[training]
epochs = 8
batch_size = 16
learning_rate = 0.001
weight_decay = 0.01
```

---

## 🚀 Execution

```powershell
python main_fastai_wandb.py --config configs/efficientnet_v2.toml
```

# Model Guide: ConvNeXt-Base (fastai + TOML + W&B)

Dedicated guide for training **ConvNeXt-Base (`convnext_base`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- Modernized pure-convolutional network with $7 \times 7$ depthwise separable convolutions, inverted bottleneck blocks, LayerNorm, and GELU activations.
- fastai transfer learning with `fine_tune(epochs=8, base_lr=0.0005, freeze_epochs=1)`.

---

## ⚙️ TOML Profile: `configs/convnext_base.toml`

```toml
[model]
framework = "fastai"
family = "convnext"
backbone = "convnext_base"
pretrained = true
freeze_epochs = 1

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.01
```

---

## 🚀 Execution

```powershell
python main_fastai_wandb.py --config configs/convnext_base.toml
```

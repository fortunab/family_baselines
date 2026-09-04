# Model Guide: ResNet50d (fastai + TOML + W&B)

Dedicated guide for training **ResNet50d (`resnet50d`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- ResNet-50 variant incorporating modified 3-convolution stem and anti-aliased downsampling blocks to preserve spatial information in histological microscopy tiles.

---

## ⚙️ TOML Profile: `configs/resnet50d.toml`

```toml
[model]
framework = "fastai"
family = "resnet"
backbone = "resnet50d"
pretrained = true
freeze_epochs = 1

[training]
epochs = 8
batch_size = 16
learning_rate = 0.001
weight_decay = 0.0001
```

---

## 🚀 Execution

```powershell
python main_fastai_wandb.py --config configs/resnet50d.toml
```

# Model Guide: Swin Transformer (fastai + TOML + W&B)

Dedicated guide for training **Swin Transformer (`swin_base_patch4_window7_224`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- Hierarchical vision transformer with shifted local windows, achieving linear computational complexity relative to image size while maintaining cross-window connectivity.

---

## ⚙️ TOML Profile: `configs/swin_transformer.toml`

```toml
[model]
framework = "fastai"
family = "swin"
backbone = "swin_base_patch4_window7_224"
pretrained = true
freeze_epochs = 1

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.05
```

---

## 🚀 Execution

```powershell
python main_fastai_wandb.py --config configs/swin_transformer.toml
```

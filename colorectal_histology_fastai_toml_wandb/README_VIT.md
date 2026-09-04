# Model Guide: Vision Transformer (ViT-Base) (fastai + TOML + W&B)

Dedicated guide for training **Vision Transformer (`vit_base_patch16_224`)** on the 8-class Colorectal Histology dataset.

---

## 🔬 Architecture Details
- Divides $224 \times 224$ histology tiles into non-overlapping $16 \times 16$ patches with learnable position embeddings and 12 multi-head self-attention transformer blocks.

---

## ⚙️ TOML Profile: `configs/vit_base.toml`

```toml
[model]
framework = "fastai"
family = "vit"
backbone = "vit_base_patch16_224"
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
python main_fastai_wandb.py --config configs/vit_base.toml
```

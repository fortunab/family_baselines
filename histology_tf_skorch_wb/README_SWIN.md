# Model Guide: Swin Transformer (skorch + TOML + W&B)

Dedicated guide for training **Swin Transformer (`swin_base_patch4_window7_224`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- Hierarchical vision transformer with shifted local windows, achieving linear computational complexity relative to image size while maintaining cross-window connectivity.
- Wrapped into `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/swin_transformer.toml`

```toml
[model]
framework = "skorch"
family = "swin"
backbone = "swin_base_patch4_window7_224"
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
python main_skorch_wandb.py --config configs/swin_transformer.toml
```

# Model Guide: ResNet50d (skorch + TOML + W&B)

Dedicated guide for training **ResNet50d (`resnet50d`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- ResNet-50 variant incorporating modified 3-convolution stem and anti-aliased downsampling blocks to preserve spatial information in histological microscopy tiles.
- Wrapped into `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/resnet50d.toml`

```toml
[model]
framework = "skorch"
family = "resnet"
backbone = "resnet50d"
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
python main_skorch_wandb.py --config configs/resnet50d.toml
```

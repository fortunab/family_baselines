# Model Guide: ConvNeXt-Base (skorch + TOML + W&B)

Dedicated guide for training **ConvNeXt-Base (`convnext_base`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- Modernized pure-convolutional network with $7 \times 7$ depthwise separable convolutions, inverted bottleneck blocks, LayerNorm, and GELU activations.
- Wrapped into `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/convnext_base.toml`

```toml
[model]
framework = "skorch"
family = "convnext"
backbone = "convnext_base"
pretrained = true

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0005
weight_decay = 0.0001
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_wandb.py --config configs/convnext_base.toml
```

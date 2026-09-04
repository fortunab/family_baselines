# Model Guide: Owkin Phikon (skorch + TOML + W&B)

Dedicated guide for training **Owkin Phikon (`owkin/phikon`)** on the 8-class Colorectal Histology dataset using `skorch`.

---

## 🔬 Architecture Details
- **Owkin Phikon** is a specialized pathology foundation model pre-trained on 40M+ histology tiles from TCGA using the iBOT self-supervised learning algorithm.
- Produces 768-dimensional latent representations for tissue feature extraction.
- Wrapped in `skorch.NeuralNetClassifier` with Scikit-Learn `.fit()` and `.predict_proba()` API.

---

## ⚙️ TOML Profile: `configs/phikon.toml`

```toml
[model]
framework = "skorch"
family = "phikon"
backbone = "owkin/phikon"
model_type = "huggingface"
pretrained = true
embedding_dim = 768

[training]
epochs = 8
batch_size = 16
learning_rate = 0.0003
weight_decay = 0.01
device = "cpu"
```

---

## 🚀 Execution

```powershell
python main_skorch_foundation.py --config configs/phikon.toml
```

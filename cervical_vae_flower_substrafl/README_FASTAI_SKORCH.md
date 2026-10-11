# FastAI & Skorch Dual-Engine Execution

This repository features dual deep learning backends—**fastai** and **skorch**—allowing direct benchmarking of high-level vision pipelines against Scikit-Learn compatible abstractions in federated setups.

---

## ⚖️ Engine Architecture Comparison

| Feature | fastai Engine | skorch Engine |
|---|---|---|
| **API Paradigm** | PyTorch fastai vision loop | Scikit-Learn `NeuralNetClassifier` wrapper |
| **Optimization** | AdamW with 1-Cycle learning rate scheduling | AdamW via standard epoch iterator |
| **FedProx Integration** | Explicit proximal penalty $\frac{\mu}{2}\|\mathbf{w} - \mathbf{w}_t\|^2$ in batch loss | Custom `get_loss()` override with proximal penalty |
| **Preprocessors** | Ingests PIL micrographs via VAE transforms | Tensors extracted via preprocessed PIL loader |
| **Flag** | `--framework fastai` | `--framework skorch` |

---

## 🏃 Commands

```powershell
# Run with fastai backend
python main_federated_flower.py --framework fastai --preprocessing vae

# Run with skorch backend
python main_federated_flower.py --framework skorch --preprocessing vae
```

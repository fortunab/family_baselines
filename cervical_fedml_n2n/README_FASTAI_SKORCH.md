# FastAI vs Skorch Execution Backends

This suite supports seamless dual-backend execution across **fastai** and **skorch** for all federated strategies and preprocessing modes.

---

## ⚖️ Backend Comparison

| Dimension | `fastai` Engine | `skorch` Engine |
|---|---|---|
| **API Paradigm** | High-level deep learning API | Scikit-Learn compatible PyTorch wrapper |
| **Optimizers** | AdamW with 1-Cycle Policy scheduling | AdamW via standard `NeuralNetClassifier` loop |
| **Transform Integration** | fastai pipeline with PILImage conversions | Tensor extraction with PIL preprocessors |
| **FedProx Support** | Native custom PyTorch loss in batch loop | Custom `get_loss()` override with proximal penalty |
| **Execution Command** | `--framework fastai` | `--framework skorch` |

---

## 🏃 Running Examples

```powershell
# Run with fastai backend
python main_federated_fedml.py --framework fastai --preprocessing n2n_cyclegan

# Run with skorch backend
python main_federated_fedml.py --framework skorch --preprocessing n2n_cyclegan
```

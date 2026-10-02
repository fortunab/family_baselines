# skorch NeuralNetClassifier Architecture & Scikit-Learn Integration

This document details the architectural integration between **skorch**, **PyTorch**, and **Flower (`flwr`)** for cervical cytology dysplasia grading.

---

## 1. What is skorch?

[skorch](https://skorch.readthedocs.io/) is a high-level library that wraps PyTorch neural networks in a **Scikit-Learn compatible API**:
- Calling `.fit(X, y)` trains the network using PyTorch optimizers and loss functions.
- Calling `.predict(X)` outputs discrete class predictions.
- Calling `.predict_proba(X)` outputs softmax class probabilities.
- Seamless compatibility with Scikit-Learn pipelines, GridSearchCV, and standard metrics.

---

## 2. Custom `SkorchCervicalClassifier` Implementation

In `src/skorch_engine.py`:
```python
class SkorchCervicalClassifier(NeuralNetClassifier):
    def __init__(
        self,
        module,
        *args,
        is_fedprox: bool = False,
        proximal_mu: float = 0.0,
        global_anchor_tensors: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__(module, *args, **kwargs)
        self.is_fedprox = is_fedprox
        self.proximal_mu = proximal_mu
        self.global_anchor_tensors = global_anchor_tensors

    def get_loss(self, y_pred, y_true, X=None, training=False):
        base_loss = super().get_loss(y_pred, y_true, X=X, training=training)
        if training and self.is_fedprox and self.global_anchor_tensors is not None:
            prox_reg = 0.0
            for w, w_t in zip(self.module_.parameters(), self.global_anchor_tensors):
                prox_reg += ((w - w_t) ** 2).sum()
            return base_loss + (self.proximal_mu / 2.0) * prox_reg
        return base_loss
```

---

## 3. Parameter Synchronization with Flower NumPyClient

Flower clients exchange weights as a list of NumPy NDArrays:
- **Extraction**: `get_model_parameters(net.module_)` iterates over `module.state_dict().items()` and returns `val.cpu().numpy()`.
- **Injection**: `set_model_parameters(net.module_, parameters)` maps NDArrays back into PyTorch tensors and updates the module state dict with `strict=True`.

---

## 4. Advantages of skorch in Federated Learning
1. **Decoupled Architecture**: Training loops are standardized and separated from framework-specific learner abstractions.
2. **Deterministic Tensor Feeding**: Data is transformed directly into memory-efficient float tensors `(X, y)`, minimizing CPU-to-GPU overhead.
3. **Built-in Callback System**: Easily extensible with early stopping, learning rate warmup, and validation scoring.

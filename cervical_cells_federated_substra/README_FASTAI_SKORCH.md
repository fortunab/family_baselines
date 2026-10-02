# Dual-Framework Engine Guide: fastai + skorch in Substra

This repository provides seamless execution across two modern machine learning paradigms for federated learning in Substra nodes:

---

## 1. The `fastai` Engine (`src/fastai_engine.py`)

- **Paradigm**: Deep learning computer vision workflows featuring the 1-Cycle training policy (`fit_one_cycle`).
- **Data Augmentation**: Robust affine transformations (random resized crop, flips, rotations).
- **FedProx Formulation**:
  $$L_{\text{FedProx}}(w; w^t) = L_{\text{CE}}(w) + \frac{\mu}{2} \sum_{l} \|w_l - w^t_l\|^2$$
- **When to Choose**: When optimizing for raw training speed, batch processing, and 1-Cycle learning rate annealing.

```powershell
python main_federated_substra.py --framework fastai --config configs/default.toml
```

---

## 2. The `skorch` Engine (`src/skorch_engine.py`)

- **Paradigm**: Scikit-Learn API wrapper for PyTorch (`NeuralNetClassifier`).
- **API Semantics**:
  - `net.fit(X, y)`
  - `net.predict_proba(X)`
  - Direct compatibility with Scikit-Learn pipelines, grid searches, and metrics.
- **Native FedProx in skorch**:
  Implemented by subclassing `NeuralNetClassifier` and overriding `get_loss()`:
  ```python
  class SkorchCervicalClassifier(NeuralNetClassifier):
      def get_loss(self, y_pred, y_true, X=None, training=False):
          loss = super().get_loss(y_pred, y_true, X=X, training=training)
          if training and self.is_fedprox and self.proximal_mu > 0.0:
              prox_term = torch.tensor(0.0, device=y_pred.device)
              for p, g in zip(self.module_.parameters(), self.global_anchor_tensors):
                  prox_term += torch.sum((p - g.to(p.device)) ** 2)
              loss += (self.proximal_mu / 2.0) * prox_term
          return loss
  ```
- **When to Choose**: When standard Scikit-Learn evaluation, serialization, and clean tabular/tensor workflows are required.

```powershell
python main_federated_substra.py --framework skorch --config configs/default.toml
```

---

## 3. Systematic Head-to-Head Comparison

Compare both engines on the exact same Substra clinical partitions and random best selected seed:

```powershell
python compare_substra_strategies.py --strategies FedAvg FedProx --frameworks fastai skorch --rounds 3
```

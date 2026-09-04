# skorch Scikit-Learn NeuralNetClassifier Guide

Deep learning pipeline powered by **[`skorch`](https://skorch.readthedocs.io/)**, bridging PyTorch vision backbones with the Scikit-Learn ecosystem.

---

## 🔬 Core skorch Features Used

1. **`NeuralNetClassifier`**:
   - Wraps PyTorch vision networks into an estimator that adheres strictly to the standard Scikit-Learn API:
     - `.fit(X, y)`
     - `.predict(X)`
     - `.predict_proba(X)`
     - `.score(X, y)`

2. **Scikit-Learn Ecosystem Compatibility**:
   - Seamless interoperability with:
     - `sklearn.model_selection.GridSearchCV` / `RandomizedSearchCV`
     - `sklearn.model_selection.cross_val_score`
     - `sklearn.pipeline.Pipeline`
     - `sklearn.metrics`

3. **Built-in skorch Callbacks**:
   - `LRScheduler` with Cosine Annealing.
   - `EpochScoring` monitoring validation accuracy.
   - `EarlyStopping` preventing overfitting on small medical datasets.

---

## 🚀 Execution Example

```powershell
python main_skorch.py --config configs/skorch_resnet.ini
```

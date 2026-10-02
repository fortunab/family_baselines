# Dual Framework Integration Guide: fastai & skorch in FedML

This suite features seamless inter-compatibility between **`fastai`** and **`skorch`** backends within a unified FedML federated architecture.

---

## 1. Comparing the Frameworks

| Characteristic | fastai Backend (`src/fastai_engine.py`) | skorch Backend (`src/skorch_engine.py`) |
| :--- | :--- | :--- |
| **API Paradigm** | High-level `Learner` & `DataBlock` | Scikit-Learn `.fit(X, y)` and `.predict_proba(X)` |
| **Optimization** | 1-Cycle Policy (`fit_one_cycle`) | Standard AdamW via PyTorch optimizers |
| **FedProx Regularization** | Custom loss function `FastAIFedProxLoss` | Overridden `get_loss()` in `SkorchCervicalClassifier` |
| **Data Ingestion** | On-the-fly PIL image pipeline (`ColReader`) | Pre-extracted Tensor batches `(N, 3, 224, 224)` |
| **Ideal Use Case** | Vision models requiring high-performance schedules | Scikit-Learn pipelines, grid searches, ensembles |

---

## 2. Using the fastai Backend

Select fastai using the `--framework fastai` CLI argument or via TOML configuration:
```powershell
python main_federated_fedml.py --config configs/fastai_convnext.toml --framework fastai
```

fastai utilizes the 1-Cycle policy to accelerate client convergence during brief local epochs, making it resilient to client data sparsity.

---

## 3. Using the skorch Backend

Select skorch using the `--framework skorch` CLI argument:
```powershell
python main_federated_fedml.py --config configs/skorch_convnext.toml --framework skorch
```

Under skorch, the client initializes a `SkorchCervicalClassifier` instance that adheres strictly to Scikit-Learn's estimator contract:
```python
net = SkorchCervicalClassifier(
    module=model,
    criterion=nn.CrossEntropyLoss,
    optimizer=optim.AdamW,
    lr=0.0003,
    max_epochs=2,
    is_fedprox=True,
    proximal_mu=1.0,
    global_anchor_tensors=global_tensors,
)
net.fit(X_client, y_client)
```

---

## 4. Benchmark Comparison

To systematically compare convergence, accuracy, and runtime across both frameworks:
```powershell
python compare_fedml_strategies.py --rounds 3 --clients 3 --subsample 20
```
This generates comparative CSV logs and a visualization plot at `results/fedml_strategy_comparison.png`.

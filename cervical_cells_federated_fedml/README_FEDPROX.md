# FedProx Proximal Regularization Guide

This guide details the mathematical foundation and implementation of **FedProx** in FedML for cervical cytology classification under severe statistical heterogeneity (non-IID).

---

## 1. The Challenge of Non-IID Hospital Partitions

In decentralized clinical environments, individual hospital sites collect unbalanced patient distributions:
- Specialized oncology centers exhibit an over-representation of severe dysplasia and carcinoma in situ.
- Routine screening outpatient clinics predominantly observe normal superficial or intermediate cells.

When training standard FedAvg under high data heterogeneity (e.g. Dirichlet $\alpha=0.5$), local model updates diverge significantly from the global objective (client drift), degrading aggregated performance.

---

## 2. FedProx Mathematical Formulation

FedProx introduces a proximal regularization term to the local client objective:

$$\min_{w} h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2$$

where:
- $F_k(w)$ is the local empirical cross-entropy loss at hospital silo $k$.
- $w^t$ represents the global weights received from the server at round $t$.
- $\mu \ge 0$ is the proximal regularization hyperparameter (e.g. $\mu=1.0$).
- $\|w - w^t\|^2 = \sum_{l} \|w_l - w^t_l\|_2^2$ penalizes local client parameters from drifting excessively far from the global consensus.

---

## 3. Implementation in FedML

In this suite, FedProx is supported across both backends:
1. **fastai**: Handled by `FastAIFedProxLoss` in `src/fastai_engine.py`, computing the Euclidean penalty against cached global parameter tensors during local backward passes.
2. **skorch**: Implemented inside `SkorchCervicalClassifier.get_loss()` in `src/skorch_engine.py`, directly modifying the PyTorch optimization loss graph before backpropagation.

---

## 4. Running FedProx Experiments

To run FedProx on Dirichlet non-IID partitions:
```powershell
python main_federated_fedml.py --config configs/fedprox_non_iid.toml --strategy FedProx
```

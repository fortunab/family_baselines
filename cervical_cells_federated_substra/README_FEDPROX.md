# FedProx Proximal Regularization Guide: Substra Cervical Cytology

Mathematical formulation and implementation of **FedProx** for heterogeneous, non-IID cervical cytology distributions.

---

## 1. Problem Formulation: Statistical Heterogeneity

In clinical cytology screening, distribution skew arises naturally:
- Tertiary oncology clinics encounter significantly higher proportions of High-Grade Squamous Intraepithelial Lesions (`05_moderate_dysplastic`, `06_severe_dysplastic`, `07_carcinoma_in_situ`).
- Community mobile vans and routine screening clinics primarily observe benign cellular morphology (`01_normal_superficiel`, `02_normal_intermediate`, `03_normal_columnar`).

This non-IID data distribution causes **client drift** when optimizing standard FedAvg:
$$\min_w \sum_{k=1}^K p_k F_k(w)$$

---

## 2. FedProx Proximal Regularization

FedProx stabilizes local updates by adding a proximal penalty term centered around the global model weights $w^t$:
$$\min_w h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2$$

Where $\mu \ge 0$ governs the trade-off between local fitting and global alignment.

---

## 3. Implementation Across Both Engines

- **In `fastai`**: Computed dynamically in the backward pass during local epochs.
- **In `skorch`**: Evaluated directly inside overridden `get_loss()` of `SkorchCervicalClassifier`.

---

## 4. Running FedProx on Non-IID Dirichlet Partitions

```powershell
python main_federated_substra.py --config configs/fedprox_non_iid.toml --framework fastai
python main_federated_substra.py --config configs/fedprox_non_iid.toml --framework skorch
```

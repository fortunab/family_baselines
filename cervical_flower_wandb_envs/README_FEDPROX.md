# FedProx & Non-IID Dirichlet Skew in Cervical Cytology

This guide explains the theoretical and practical underpinnings of handling non-IID data distribution in distributed clinical Pap smear screening.

---

## 1. Clinical Non-IID Skew in Cervical Screening

In real-world cervical cancer screening:
- **Tertiary Oncology Centers**: Receive a disproportionate fraction of severe dysplasia and carcinoma in situ cases (`05_moderate_dysplastic`, `06_severe_dysplastic`, `07_carcinoma_in_situ`).
- **Primary Care Clinics / Screening Vans**: Primarily observe benign and normal cytology (`01_normal_superficiel`, `02_normal_intermediate`, `03_normal_columnar`).

This creates extreme label distribution skew ($\mathcal{P}_i(y) \neq \mathcal{P}_j(y)$) across participating hospital nodes.

---

## 2. Dirichlet Distribution Modeling ($\alpha$)

To simulate real-world clinical heterogeneity, the dataset is partitioned across clients using a Dirichlet distribution $\text{Dir}(\alpha)$:
- $\alpha \to \infty$: Uniform IID distribution across all hospital sites.
- $\alpha = 0.5$: Moderate-to-severe non-IID skew (standard clinical benchmark).
- $\alpha = 0.1$: Pathological non-IID skew where individual clients may only observe 1 or 2 classes.

In `configs/fedprox_non_iid.toml`:
```toml
[dataset]
non_iid = true
dirichlet_alpha = 0.5
```

---

## 3. FedProx Proximal Regularization

Under non-IID skew, local client models overfit to their local class distribution, causing client drift that destabilizes FedAvg.
FedProx (Li et al., 2020) introduces a proximal term to the local optimization objective:

$$\min_{w} h_k(w; w^t) = F_k(w) + \frac{\mu}{2} \|w - w^t\|^2$$

Where:
- $w^t$: Global model weights received from the server at round $t$.
- $w$: Local model parameters during client training.
- $\mu$: Proximal regularization coefficient (typically $\mu \in [0.01, 1.0]$).

### Effect of $\mu$:
- Prevents local models from drifting too far from global consensus.
- Stabilizes convergence on highly skewed client partitions.
- Tolerates partial local client work and heterogeneous local compute capabilities.

# FedML Federated Learning Engine for Cervical Cytology

This module implements a decoupled cross-silo federated learning architecture based on the **FedML** paradigm (`ClientTrainer` and `ServerAggregator`).

---

## 🏛️ Federated Architecture

Decentralized hospital silos train local vision backbones on private patient cytology datasets without sharing raw patient micrographs with external entities. Only parameter weights ($\mathbf{w}_k$) and sample counts ($n_k$) are shared with the central server.

### Supported Federated Strategies

1. **FedAvg (McMahan et al., 2017)**:
   $$\mathbf{w}_{t+1} = \sum_{k=1}^K \frac{n_k}{N} \mathbf{w}_{t+1}^k$$
   Standard sample-weighted averaging across all active client silos.

2. **FedProx (Li et al., 2020)**:
   $$\mathcal{L}_{\text{local}}(\mathbf{w}; \mathbf{w}_t) = \mathcal{L}_{\text{task}}(\mathbf{w}) + \frac{\mu}{2} \|\mathbf{w} - \mathbf{w}_t\|_2^2$$
   Adds a proximal regularization penalty constraint ($\mu$) preventing local client models from drifting too far from global weights when partitions exhibit severe class distribution non-IIDness (Dirichlet $\alpha \le 0.5$).

3. **FedAdam (Reddi et al., 2021)**:
   Treats server weight updates as an adaptive optimization step:
   $$\Delta_t = \sum_{k=1}^K \frac{n_k}{N} (\mathbf{w}_{t+1}^k - \mathbf{w}_t)$$
   $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \Delta_t$$
   $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \Delta_t^2$$
   $$\mathbf{w}_{t+1} = \mathbf{w}_t + \eta \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \tau}$$

---

## 💻 Code Abstractions

- `FedMLClientCervicalTrainer`: Manages local hospital silo state, training iterations, local loss tracking, and preprocessing dispatch (`src/fedml_client.py`).
- `FedMLCervicalServerAggregator`: Collects client weights, executes aggregation strategies, updates global model, evaluates on centralized holdout test partition, and logs to W&B (`src/fedml_server.py`).

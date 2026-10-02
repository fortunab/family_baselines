# FedML Cross-Silo Architecture Guide

This guide details the FedML cross-silo simulation architecture implemented for 7-class cervical cytology classification.

---

## 1. Architectural Overview

FedML provides a structured API separating client-side local computation from server-side global coordination:

```
[Coordinator / Aggregator Node]
   FedMLCervicalServerAggregator
       ├── Distribute Global Weights (w_t)
       ├── Collect Local Updates from Clients (w_k^{t+1}, n_k)
       ├── Strategy Aggregator (FedAvg / FedProx / FedAdam)
       └── Centralized Holdout Evaluation (15% Unseen Test Set)
              ▲                ▲                ▲
              │                │                │
              ▼                ▼                ▼
       [Hospital Silo 1] [Hospital Silo 2] [Hospital Silo 3] ...
         FedMLClient      FedMLClient      FedMLClient
        CervicalTrainer  CervicalTrainer  CervicalTrainer
          (fastai/         (fastai/         (fastai/
           skorch)          skorch)          skorch)
```

---

## 2. FedML Client Trainer (`src/fedml_client.py`)

The `FedMLClientCervicalTrainer` class encapsulates individual hospital clinics:
- Ingests global weights received from coordinator.
- Manages local private data partitions (isolated strictly to the client).
- Dispatches training to either `fastai` (1-Cycle policy, FedProx loss) or `skorch` (`SkorchCervicalClassifier`, FedProx penalty, `.fit()`).
- Extracts updated parameter arrays and reports local sample counts.

---

## 3. FedML Server Aggregator (`src/fedml_server.py`)

The `FedMLCervicalServerAggregator` class coordinates the federated rounds:
- **FedAvg**:
  $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}$$
- **FedProx**:
  Aggregates client weights trained under proximal regularization $L_{prox}(w; w_t) = L_{ce}(w) + \frac{\mu}{2} \|w - w_t\|^2$.
- **FedAdam**:
  Server-side adaptive momentum optimizer:
  $$\Delta_t = \sum_{k=1}^K \frac{n_k}{N} (w_k^{t+1} - w_t)$$
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \Delta_t^2$$
  $$w_{t+1} = w_t + \eta \frac{m_t}{\sqrt{v_t} + \tau}$$

---

## 4. Centralized Holdout Evaluation

After every communication round, the server evaluates the updated global weights on a dedicated 15% stratified centralized holdout set:
- Ensures real-time tracking of global generalization performance.
- Computes Accuracy, Balanced Accuracy, Macro F1, Weighted F1, AUC-ROC, and Cohen's Kappa.
- Telemeters metrics directly to Weights & Biases (W&B) and records to local JSON/CSV logs.

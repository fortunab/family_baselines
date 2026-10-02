# Substra Federated Learning Architecture: Cervical Cytology

This suite implements the enterprise **Substra** privacy-preserving federated architecture for multi-center cervical cancer screening.

---

## 1. Substra Asset Hierarchy

In Substra, computations are structured into formal distributed assets:

```
[Clinical Center 0]              [Clinical Center 1]              [Coordinator Node]
+----------------------+        +----------------------+        +----------------------+
| SubstraCervicalOpener|        | SubstraCervicalOpener|        | Central Test Set     |
| (Local Data Only)    |        | (Local Data Only)    |        | (15% Unseen Holdout) |
+----------+-----------+        +----------+-----------+        +----------+-----------+
           |                               |                               |
           v                               v                               v
+----------+-----------+        +----------+-----------+        +----------+-----------+
| SubstraCervicalAlgo  |        | SubstraCervicalAlgo  |        | Global Aggregator    |
| (fastai or skorch)   |        | (fastai or skorch)   |        | (FedAvg / FedProx)   |
+----------+-----------+        +----------+-----------+        +----------+-----------+
           |                               |                               ^
           +-----------------------+-------+-------------------------------+
                                   | (Weights Only - No Patient Data)
                                   v
                      Updated Global Model Checkpoint
```

---

## 2. Core Substra Assets

### 1. `SubstraCervicalOpener` (`src/substra_opener.py`)
- Extends `substratools.Opener`.
- Manages local filesystem discovery and image loading for an organization node.
- Provides `fake_data()` generating synthetic 7-class records for Substra debug runs without accessing patient data.

### 2. `SubstraCervicalAlgo` (`src/substra_algo.py`)
- Discrete function asset executed in a sandbox/Docker container at each hospital node.
- Ingests global weights, executes local training (`fastai` or `skorch`), and exports updated parameter weights.
- Guarantees zero clinical data leakage.

### 3. `SubstraComputePlanOrchestrator` (`src/substra_orchestrator.py`)
- Coordinates the Directed Acyclic Graph (DAG) of federated tasks across aggregation rounds.
- Performs centralized validation on the coordinator node holding the stratified 15% holdout test partition.

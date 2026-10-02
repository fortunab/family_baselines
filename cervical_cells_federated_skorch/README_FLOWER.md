# Flower (flwr) Federated Learning Architecture

This document describes how Flower coordinates distributed training across simulated cytology clinic clients using `skorch`.

---

## 1. Federated Protocol Flow

```
                      [ Flower Central Server ]
                 Aggregator: FedAvg / FedProx / FedAdam
                                 |
              +------------------+------------------+
              |                                     |
       [ Round Dispatch ]                   [ Holdout Evaluation ]
     Global Weights $\theta^t$               15% Central Holdout Set
              |                                     |
    +---------+---------+                           |
    |                   |                           |
[ Client 0 ]       [ Client 1 ] ... [ Client K ]    |
 skorch .fit()      skorch .fit()    skorch .fit()  |
 Local Partitions   Local Partitions Local Partitions
    |                   |                           |
    +---------+---------+                           |
              |                                     |
       [ Aggregation ]                              |
   $\theta^{t+1} = \sum \frac{n_k}{n} \theta_k^{t+1}$+
```

---

## 2. Flower Client Lifecycle (`FlwrSkorchCervicalClient`)

1. **`__init__`**:
   Instantiates model architecture with `pretrained=False` (uninitialized weights). The global weights dispatched by the server will populate the model before any forward pass.
2. **`fit(parameters, config)`**:
   - Updates local model with server `parameters`.
   - Executes `net.fit(X_train, y_train)` for $E$ local epochs using AdamW and CrossEntropyLoss (with optional FedProx penalty).
   - Extracts updated weights via `get_model_parameters(self.model)`.
   - Returns `(updated_weights, num_samples, {"train_loss": loss})`.
3. **`evaluate(parameters, config)`**:
   - Synchronizes model with global parameters.
   - Computes local validation loss and accuracy.

---

## 3. Concurrency Bounding on Windows

To ensure process stability and eliminate Ray worker `ActorDiedError` (10054) on multi-core systems:
```python
total_cpus = os.cpu_count() or 4
client_cpus = max(1, total_cpus // 2)

flwr.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=server_config,
    strategy=strategy,
    client_resources={"num_cpus": client_cpus, "num_gpus": 0.0},
)
```
This restricts the Ray Virtual Client Engine to at most **2 simultaneous client actors**, avoiding thread exhaustion and memory pressure.

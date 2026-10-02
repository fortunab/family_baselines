# Windows Native Deployment & Execution Guide

This document covers native Windows execution details, PyTorch CUDA compatibility, OpenMP threading considerations, and Ray concurrency management for the Cervical Cytology skorch Suite.

---

## 1. Native Windows Considerations

### 1.1 Intel OpenMP Multiple Runtimes Error Fix (`libiomp5md.dll`)
When PyTorch, NumPy, and OpenCV are loaded simultaneously on Windows, Intel OpenMP can throw:
`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

**Built-in Fix**:
`src/__init__.py` automatically injects:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
No manual intervention is required.

### 1.2 Ray & Flower Concurrency Bounding on Windows
On a 16-core system, Flower's default `client_resources = {"num_cpus": 1}` calculation spawns 16 actors simultaneously, exhausting RAM and causing `ActorDiedError: Worker unexpectedly exits with connection error code 10054`.

**Built-in Solution in `main_federated_skorch.py`**:
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
This restricts the actor pool to at most 2 concurrent client workers, keeping memory usage minimal.

---

## 2. Windows PowerShell Execution

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Fast dry-run verification
python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb

# Full training
python main_federated_skorch.py --config configs/default.toml
```

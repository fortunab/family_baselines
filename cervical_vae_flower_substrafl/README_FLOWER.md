# Flower (flwr) Federated Learning Architecture

This suite integrates Flower's decoupled federated learning ecosystem for digital cervical cytology.

---

## 🏛️ Flower Components

1. **`FlowerCervicalClient` (`src/flower_client.py`)**:
   - Inherits `flwr.client.NumPyClient`.
   - Manages local client state, dataset partitions, and VAE preprocessing.
   - Dispatches local training to either `fastai` or `skorch`.
2. **`create_flower_strategy` (`src/flower_server.py`)**:
   - Instantiates federated strategies: **FedAvg**, **FedProx**, or **FedAdam**.
   - FedProx incorporates proximal parameter penalty $\mu$ for Non-IID Dirichlet partitions.
   - FedAdam applies server-side first and second momentum adaptive updates.
3. **`flwr.simulation.start_simulation`**:
   - Runs cross-silo federated communication rounds.
   - Evaluates aggregated global model on the 15% centralized holdout test partition after each round.

---

## 🏃 Running Flower Experiments

```powershell
# Default Flower with fastai and VAE
python main_federated_flower.py --preprocessing vae --framework fastai

# Flower with skorch and FedProx
python main_federated_flower.py --preprocessing vae --framework skorch --strategy FedProx
```

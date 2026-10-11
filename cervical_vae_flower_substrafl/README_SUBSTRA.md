# Substra Federated Learning Architecture

Substra provides an enterprise-grade, privacy-preserving distributed learning framework tailored for multi-institutional healthcare consortia.

---

## 🏛️ Substra Assets

1. **Substra Opener (`SubstraCervicalOpener`)**:
   - Manages ingestion of private patient single-cell patches and manifests inside each hospital silo node.
   - Enforces data isolation: raw micrographs never leave the organization.
2. **Substra Algo (`SubstraCervicalAlgo`)**:
   - Defines the compute task executed within the organization boundary.
   - Applies the local VAE preprocessor and trains the vision model using fastai (AdamW) or skorch.
3. **Substra Strategy (`SubstraFederatedStrategy`)**:
   - Aggregates updated weights across participating organization nodes using FedAvg, FedProx, or FedAdam.
4. **Substra Platform (`SubstraSimulatedPlatform`)**:
   - Coordinates multi-round communication, registers nodes, and evaluates the global model on the centralized holdout test partition.

---

## 🏃 Running Substra Experiments

```powershell
# Default Substra with fastai and VAE
python main_federated_substra.py --preprocessing vae --framework fastai

# Substra with skorch and FedAdam
python main_federated_substra.py --preprocessing vae --framework skorch --strategy FedAdam
```

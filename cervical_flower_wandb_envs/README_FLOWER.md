# Flower (flwr) Federated Architecture & Design

This guide details the integration architecture between **Flower**, **fastai**, and the **Herlev Cervical Cytology** dataset.

---

## 1. Architectural Overview

```
                                [ Flower Server ]
                             Strategy: FedAvg / FedProx
                                       |
                       +---------------+---------------+
                       |                               |
                [ Global Weights ]             [ Centralized Test Set ]
                 Aggregated $\theta_t$          15% Unseen Holdout
                       |
        +--------------+--------------+
        |                             |
 [ Client 0 (Site A) ]         [ Client 1 (Site B) ] ... [ Client K ]
  - fastai Learner              - fastai Learner
  - Local Cervical Images       - Local Cervical Images
  - Dirichlet Skew              - Dirichlet Skew
```

---

## 2. Flower Client Implementation (`FlwrFastaiCervicalClient`)

The client inherits from `flwr.client.NumPyClient`:
- **`get_parameters(config)`**: Extracts model weights as a list of NumPy NDArrays.
- **`set_parameters(parameters)`**: Loads updated global weights into the PyTorch backbone.
- **`fit(parameters, config)`**:
  1. Synchronizes local model with global parameters.
  2. Trains for $E$ local epochs using fastai 1-Cycle policy (`learn.fit_one_cycle`).
  3. In FedProx mode, computes proximal loss penalty:
     $$\mathcal{L}_{\text{prox}} = \frac{\mu}{2} \|\theta - \theta^t\|^2$$
  4. Returns updated weights and sample count.
- **`evaluate(parameters, config)`**: Evaluates local loss and accuracy on local validation fold.

---

## 3. Server Evaluation & Aggregation

The server executes a centralized evaluation function after each aggregation round:
- Tests the newly aggregated global model against the unseen **15% Centralized Holdout Set**.
- Computes multi-class metrics: Accuracy, Macro Precision, Recall, Macro F1, Balanced Accuracy, and Multi-Class AUC (One-vs-Rest).
- Streams metrics directly to Weights & Biases (W&B) and saves confusion matrix plots.

---

## 4. Supported Strategies

1. **FedAvg**: Standard Federated Averaging (McMahan et al., 2017).
2. **FedProx**: Handles client heterogeneity and non-IID drift via proximal regularization parameter $\mu$.
3. **FedAdam**: Server-side adaptive optimizer with momentum and learning rate damping.

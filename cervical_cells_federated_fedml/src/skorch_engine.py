"""
skorch Local Training Engine for Decentralized FedML Hospital Clients.
Provides Scikit-Learn API (.fit, .predict_proba) and FedProx proximal regularization in FedML.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNetClassifier
from torch import nn, optim

from src.dataset import extract_tensors_from_df
from src.fastai_engine import get_model_parameters


class SkorchCervicalClassifier(NeuralNetClassifier):
    """
    FedML skorch NeuralNetClassifier with native FedProx proximal regularization.
    Integrates Scikit-Learn API into privacy-preserving FedML hospital client silos.
    """

    def __init__(
        self,
        module,
        *args,
        is_fedprox: bool = False,
        proximal_mu: float = 0.0,
        global_anchor_tensors: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        super().__init__(module, *args, **kwargs)
        self.is_fedprox = is_fedprox
        self.proximal_mu = proximal_mu
        self.global_anchor_tensors = global_anchor_tensors

    def get_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X: Any = None,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Computes base cross-entropy loss plus FedML FedProx proximal penalty:
        L(w; w^t) = L_ce(w) + (mu / 2) * sum_l ||w_l - w^t_l||^2
        """
        loss = super().get_loss(y_pred, y_true, X=X, training=training)

        if training and self.is_fedprox and self.proximal_mu > 0.0 and self.global_anchor_tensors:
            prox_term = torch.tensor(0.0, device=y_pred.device)
            current_params = [p for p in self.module_.parameters()]
            for p, g in zip(current_params, self.global_anchor_tensors, strict=False):
                if p.shape == g.shape:
                    prox_term = prox_term + torch.sum((p - g.to(p.device)) ** 2)
            loss = loss + (self.proximal_mu / 2.0) * prox_term

        return loss


def resolve_device_str(requested_device: str | None = None) -> str:
    """Safely resolves device string to 'cuda' only if available, otherwise 'cpu'."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if requested_device in ["cpu", "cuda"]:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def local_train_skorch(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training loop using skorch on an individual hospital silo.
    Extracts image tensors, initializes SkorchCervicalClassifier, and fits locally.
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    batch_size = int(cfg.get("batch_size", 16))
    local_epochs = int(cfg.get("local_epochs", 2))
    local_lr = float(cfg.get("local_lr", 0.0003))
    device = resolve_device_str(cfg.get("device"))
    strategy = cfg.get("strategy", "FedAvg")
    mu = float(cfg.get("proximal_mu", 1.0)) if strategy == "FedProx" else 0.0

    # Extract tensors for skorch fit
    x_train, y_train = extract_tensors_from_df(client_df, img_size=224)

    # Convert global weights to tensors for FedProx anchor if needed
    global_tensors = None
    if strategy == "FedProx" and global_model_weights is not None:
        target_device = torch.device(device)
        global_tensors = [torch.from_numpy(w).to(target_device) for w in global_model_weights]

    net = SkorchCervicalClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        lr=local_lr,
        max_epochs=local_epochs,
        batch_size=batch_size,
        device=device,
        train_split=None,  # All local data used for training
        verbose=0,
        is_fedprox=(strategy == "FedProx"),
        proximal_mu=mu,
        global_anchor_tensors=global_tensors,
    )

    # Fit using standard Scikit-Learn API
    net.fit(x_train, y_train)

    # Retrieve last epoch training loss from net history
    try:
        final_loss = float(net.history[-1, "train_loss"])
    except (IndexError, KeyError):
        final_loss = 0.0

    updated_weights = get_model_parameters(net.module_)
    return updated_weights, final_loss, len(client_df)


def evaluate_model_on_test_set_skorch(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates global model on the centralized holdout test set using skorch wrapper.
    Returns (y_true, y_prob, avg_loss).
    """
    device = resolve_device_str(cfg.get("device"))
    x_test, y_test = extract_tensors_from_df(test_df, img_size=224)

    net = SkorchCervicalClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        device=device,
        train_split=None,
        verbose=0,
    )
    # Initialize net without training
    net.initialize()

    # Predict probabilities via Scikit-Learn API
    y_prob = net.predict_proba(x_test)
    y_true = y_test.cpu().numpy()

    # Calculate cross-entropy loss
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = torch.from_numpy(y_prob).log()
        loss = float(loss_fn(logits, y_test).item())

    return y_true, y_prob, loss

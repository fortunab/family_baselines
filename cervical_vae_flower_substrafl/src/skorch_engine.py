"""
skorch NeuralNetClassifier Engine for Cervical Cytology.
Provides Scikit-Learn API interoperability with PyTorch models and FedProx support.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNetClassifier
from torch import nn, optim

from src.dataset import extract_tensors_from_df
from src.fastai_engine import get_model_parameters, resolve_device


class SkorchCervicalClassifier(NeuralNetClassifier):
    """
    Custom skorch wrapper for cervical cytology classification.
    Injects proximal regularization term during FedProx training.
    """

    def __init__(
        self,
        *args: Any,
        is_fedprox: bool = False,
        proximal_mu: float = 1.0,
        global_anchor_tensors: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.is_fedprox = is_fedprox
        self.proximal_mu = proximal_mu
        self.global_anchor_tensors = global_anchor_tensors

    def get_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Calculates loss with optional FedProx proximal penalty term."""
        base_loss = super().get_loss(y_pred, y_true, *args, **kwargs)

        if not self.is_fedprox or not self.global_anchor_tensors:
            return base_loss

        prox_loss = torch.tensor(0.0, device=y_pred.device)
        for w, w_t in zip(self.module_.parameters(), self.global_anchor_tensors, strict=False):
            prox_loss += torch.sum((w - w_t) ** 2)

        return base_loss + (self.proximal_mu / 2.0) * prox_loss


def resolve_device_str(device_cfg: str | None = None) -> str:
    """Returns resolved device as string compatible with skorch."""
    return str(resolve_device(device_cfg))


def local_train_skorch(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
    preprocessor: Any = None,
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

    x_train, y_train = extract_tensors_from_df(client_df, img_size=224, preprocessor=preprocessor)

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
        train_split=None,
        verbose=0,
        is_fedprox=(strategy == "FedProx"),
        proximal_mu=mu,
        global_anchor_tensors=global_tensors,
    )

    net.fit(x_train, y_train)

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
    preprocessor: Any = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates global model on centralized holdout test set using skorch wrapper.
    Returns (y_true, y_prob, avg_loss).
    """
    device = resolve_device_str(cfg.get("device"))
    x_test, y_test = extract_tensors_from_df(test_df, img_size=224, preprocessor=preprocessor)

    net = SkorchCervicalClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        device=device,
        train_split=None,
        verbose=0,
    )
    net.initialize()

    y_prob = net.predict_proba(x_test)
    y_true = y_test.cpu().numpy()

    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = torch.from_numpy(y_prob).log()
        loss = float(loss_fn(logits, y_test).item())

    return y_true, y_prob, loss

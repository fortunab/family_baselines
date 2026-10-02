"""
skorch Local Training Engine for Substra Nodes on Cervical Cytology.
Provides Scikit-Learn API (.fit, .predict_proba) and FedProx proximal regularization in Substra.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNetClassifier
from torch import nn, optim

from src.dataset import extract_tensors_from_df
from src.fastai_engine import (
    build_cervical_model,
    get_model_parameters,
)


class SkorchCervicalClassifier(NeuralNetClassifier):
    """
    Substra skorch NeuralNetClassifier with native FedProx proximal regularization.
    Integrates Scikit-Learn API into privacy-preserving Substra organization nodes.
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
        Computes base cross-entropy loss plus Substra FedProx proximal penalty:
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


def build_skorch_classifier(
    arch_name: str,
    num_classes: int,
    lr: float,
    max_epochs: int,
    batch_size: int,
    device: str,
    dropout: float = 0.2,
    is_fedprox: bool = False,
    proximal_mu: float = 0.0,
    global_weights: list[np.ndarray] | None = None,
) -> SkorchCervicalClassifier:
    """Factory creating configured SkorchCervicalClassifier instance for Substra node."""
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    anchor_tensors = None
    if is_fedprox and global_weights is not None:
        anchor_tensors = [torch.tensor(w, device=device_obj) for w in global_weights]

    net = SkorchCervicalClassifier(
        module=build_cervical_model,
        module__arch_name=arch_name,
        module__num_classes=num_classes,
        module__pretrained=False,
        module__dropout=dropout,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=1e-4,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device_obj,
        train_split=None,  # Use all local clinic data for client training
        verbose=0,
        is_fedprox=is_fedprox,
        proximal_mu=proximal_mu,
        global_anchor_tensors=anchor_tensors,
    )
    return net


def local_train_skorch(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training in a Substra organization node using skorch.
    Returns: (updated_weights, train_loss, num_samples)
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    epochs = int(cfg.get("local_epochs", 2))
    lr = float(cfg.get("local_lr", 0.0003))
    batch_size = min(int(cfg.get("batch_size", 32)), max(2, len(client_df)))
    img_size = int(cfg.get("image_size", 224))
    proximal_mu = float(cfg.get("proximal_mu", 1.0))
    is_fedprox = (cfg.get("strategy", "FedAvg") == "FedProx") and (proximal_mu > 0.0)

    # 1. Extract image tensors and labels
    X, y = extract_tensors_from_df(client_df, img_size=img_size, augment=True)
    if len(X) == 0:
        return get_model_parameters(model), 0.0, 0

    # 2. Build skorch classifier wrapping local model
    arch_name = cfg.get("architecture", "convnext_small")
    num_classes = int(cfg.get("num_classes", 7))
    dropout = float(cfg.get("dropout_rate", 0.2))

    net = build_skorch_classifier(
        arch_name=arch_name,
        num_classes=num_classes,
        lr=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        device=str(device),
        dropout=dropout,
        is_fedprox=is_fedprox,
        proximal_mu=proximal_mu,
        global_weights=global_model_weights,
    )

    # Initialize skorch internal module with current weights
    net.initialize()
    if hasattr(net, "module_"):
        net.module_.load_state_dict(model.state_dict())

    # 3. Fit via Scikit-Learn API: net.fit(X, y)
    net.fit(X, y)

    # Synchronize updated weights back to model
    if hasattr(net, "module_"):
        model.load_state_dict(net.module_.state_dict())

    history = getattr(net, "history_", None)
    train_loss = 0.0
    if history and len(history) > 0:
        last_epoch_hist = history[-1]
        train_loss = float(last_epoch_hist.get("train_loss", 0.0))

    updated_weights = get_model_parameters(model)
    return updated_weights, train_loss, len(client_df)


def evaluate_skorch_on_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluates skorch model on centralized holdout test DataFrame."""
    from src.fastai_engine import evaluate_model_on_test_set

    return evaluate_model_on_test_set(model, test_df, cfg)

"""
Flower (flwr) NumPyClient Implementation for Cervical Cytology with VAE Preprocessing.
Supports fastai and skorch execution backends in decentralized cross-silo simulation.
"""

from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from torch import nn

from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    get_model_parameters,
    local_train_fastai,
    set_model_parameters,
)
from src.preprocessing import get_preprocessor
from src.skorch_engine import (
    evaluate_model_on_test_set_skorch,
    local_train_skorch,
)


class FlowerCervicalClient(fl.client.NumPyClient):
    """
    Flower NumPyClient representing an individual hospital client silo.
    Manages local model training, parameter extraction, and local testing,
    applying configured VAE or baseline preprocessing.
    """

    def __init__(
        self,
        client_id: int,
        client_df: pd.DataFrame,
        cfg: dict[str, Any],
    ):
        self.client_id = client_id
        self.client_df = client_df
        self.cfg = cfg
        self.num_classes = int(cfg.get("num_classes", 7))
        self.arch_name = cfg.get("architecture", "convnext_small")
        self.framework = cfg.get("framework", "fastai").lower()
        self.preprocessing_mode = cfg.get("preprocessing", "vae").lower()

        self.model: nn.Module = build_cervical_model(
            arch_name=self.arch_name,
            num_classes=self.num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg.get("dropout_rate", 0.2)),
        )

        self.preprocessor = get_preprocessor(self.preprocessing_mode, cfg)

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        """Returns local model weights as a list of NumPy arrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Injects global model weights into local model."""
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Executes local client training round and returns updated weights."""
        self.set_parameters(parameters)

        if self.framework == "skorch":
            updated_weights, loss, num_samples = local_train_skorch(
                model=self.model,
                client_df=self.client_df,
                cfg=self.cfg,
                global_model_weights=parameters,
                preprocessor=self.preprocessor,
            )
        else:
            updated_weights, loss, num_samples = local_train_fastai(
                model=self.model,
                client_df=self.client_df,
                cfg=self.cfg,
                global_model_weights=parameters,
                preprocessor=self.preprocessor,
            )

        metrics = {"train_loss": float(loss), "client_id": self.client_id}
        return updated_weights, num_samples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any] | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluates model weights on local client data."""
        self.set_parameters(parameters)

        if self.framework == "skorch":
            y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
                model=self.model,
                test_df=self.client_df,
                cfg=self.cfg,
                preprocessor=self.preprocessor,
            )
        else:
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                model=self.model,
                test_df=self.client_df,
                cfg=self.cfg,
                preprocessor=self.preprocessor,
            )

        if len(y_true) > 0 and len(y_prob) > 0:
            preds = np.argmax(y_prob, axis=1)
            acc = float(np.mean(preds == y_true))
        else:
            acc = 0.0

        return float(test_loss), len(self.client_df), {"accuracy": acc, "loss": float(test_loss)}

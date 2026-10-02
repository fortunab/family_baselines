"""
FedML Client Trainer Implementation for Cervical Cytology.
Conforms to FedML ClientTrainer abstraction for cross-silo hospital clients,
supporting both fastai and skorch execution backends.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    get_model_parameters,
    local_train_fastai,
    set_model_parameters,
)
from src.skorch_engine import (
    evaluate_model_on_test_set_skorch,
    local_train_skorch,
)


class FedMLClientCervicalTrainer:
    """
    FedML Client Trainer representing an individual hospital client silo.
    Manages local model training, parameter extraction, and local testing,
    seamlessly dispatching to either fastai or skorch backends.
    """

    def __init__(self, client_id: int, cfg: dict[str, Any]):
        self.client_id = client_id
        self.cfg = cfg
        self.num_classes = int(cfg.get("num_classes", 7))
        self.arch_name = cfg.get("architecture", "convnext_small")
        self.framework = cfg.get("framework", "fastai").lower()

        self.model: nn.Module = build_cervical_model(
            arch_name=self.arch_name,
            num_classes=self.num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg.get("dropout_rate", 0.2)),
        )

    def get_model_params(self) -> list[np.ndarray]:
        """Returns the serialized model parameters as a list of NumPy arrays."""
        return get_model_parameters(self.model)

    def set_model_params(self, model_parameters: list[np.ndarray]) -> None:
        """Injects model parameters into the local PyTorch model."""
        set_model_parameters(self.model, model_parameters)

    def train(
        self,
        train_data: pd.DataFrame,
        global_model_weights: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], float, int]:
        """
        Executes local client training at this hospital silo:
        1. Ingests global weights
        2. Fits local model using configured framework (fastai or skorch)
        3. Returns updated weights, loss, and sample count.
        """
        if global_model_weights is not None:
            self.set_model_params(global_model_weights)

        if self.framework == "skorch":
            updated_weights, loss, num_samples = local_train_skorch(
                model=self.model,
                client_df=train_data,
                cfg=self.cfg,
                global_model_weights=global_model_weights,
            )
        else:
            updated_weights, loss, num_samples = local_train_fastai(
                model=self.model,
                client_df=train_data,
                cfg=self.cfg,
                global_model_weights=global_model_weights,
            )

        return updated_weights, loss, num_samples

    def test(
        self,
        test_data: pd.DataFrame,
    ) -> tuple[float, float]:
        """Evaluates the local model on a validation or test set."""
        if self.framework == "skorch":
            y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
                model=self.model,
                test_df=test_data,
                cfg=self.cfg,
            )
        else:
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                model=self.model,
                test_df=test_data,
                cfg=self.cfg,
            )

        if len(y_true) > 0 and len(y_prob) > 0:
            preds = np.argmax(y_prob, axis=1)
            test_acc = float(np.mean(preds == y_true))
        else:
            test_acc = 0.0

        return test_loss, test_acc

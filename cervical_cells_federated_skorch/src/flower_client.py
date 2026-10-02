"""
Flower (flwr) NumPyClient Implementation for Cervical Cytology with skorch.
Wraps skorch NeuralNetClassifier within the Flower Federated Learning Client Protocol.
"""

from collections.abc import Callable
from typing import Any

import flwr
import pandas as pd
import torch
from flwr.common import Scalar
from torch import nn

from src.skorch_engine import (
    build_cervical_model,
    evaluate_skorch_on_test_set,
    get_model_parameters,
    local_train_skorch,
    set_model_parameters,
)


class FlwrSkorchCervicalClient(flwr.client.NumPyClient):
    """
    Flower Client representing an individual cytology screening clinic.
    Executes local model fine-tuning with skorch Estimator API and exchanges serialized weights.
    """

    def __init__(
        self,
        cid: int,
        client_df: pd.DataFrame,
        cfg: dict[str, Any],
    ):
        self.cid = cid
        self.client_df = client_df
        self.cfg = cfg
        self.arch_name = cfg.get("architecture", "convnext_small")
        self.num_classes = int(cfg.get("num_classes", 7))

        if not torch.cuda.is_available():
            torch.set_num_threads(2)

        # Clients instantiate uninitialized weights; global server weights overwrite on fit()
        self.model: nn.Module = build_cervical_model(
            arch_name=self.arch_name,
            num_classes=self.num_classes,
            pretrained=False,
            dropout=float(cfg.get("dropout_rate", 0.2)),
        )

    def get_parameters(self, config: dict[str, Scalar]) -> list[Any]:
        """Extracts local model weights as NumPy ndarrays for Flower server."""
        return get_model_parameters(self.model)

    def fit(
        self,
        parameters: list[Any],
        config: dict[str, Scalar],
    ) -> tuple[list[Any], int, dict[str, Scalar]]:
        """
        Ingests global model weights and performs local skorch training.
        Returns: (updated_weights, num_samples, metrics_dict)
        """
        set_model_parameters(self.model, parameters)

        updated_weights, loss, num_samples = local_train_skorch(
            model=self.model,
            client_df=self.client_df,
            cfg=self.cfg,
            global_model_weights=parameters,
        )

        return (
            updated_weights,
            num_samples,
            {"train_loss": float(loss), "client_id": int(self.cid)},
        )

    def evaluate(
        self,
        parameters: list[Any],
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluates model weights on local clinic partition."""
        set_model_parameters(self.model, parameters)

        y_true, y_prob, loss = evaluate_skorch_on_test_set(
            model=self.model,
            test_df=self.client_df,
            cfg=self.cfg,
        )
        acc = float((y_prob.argmax(axis=1) == y_true).mean()) if len(y_true) > 0 else 0.0

        return (
            float(loss),
            len(y_true),
            {"accuracy": acc, "client_id": int(self.cid)},
        )


def make_client_fn(
    client_dfs: list[pd.DataFrame],
    cfg: dict[str, Any],
) -> Callable[..., flwr.client.Client]:
    """Flower client factory creating FlwrSkorchCervicalClient instances."""

    def client_fn(cid_or_context: Any) -> flwr.client.Client:
        if isinstance(cid_or_context, (str, int)):
            cid_str = str(cid_or_context)
        elif hasattr(cid_or_context, "node_config") and isinstance(cid_or_context.node_config, dict):
            cid_str = str(cid_or_context.node_config.get("partition-id", 0))
        elif hasattr(cid_or_context, "partition_id"):
            cid_str = str(cid_or_context.partition_id)
        else:
            cid_str = "0"

        try:
            cid_int = int(cid_str) % len(client_dfs)
        except (ValueError, TypeError):
            cid_int = 0

        c_df = client_dfs[cid_int]
        numpy_client = FlwrSkorchCervicalClient(cid=cid_int, client_df=c_df, cfg=cfg)
        return numpy_client.to_client()

    return client_fn

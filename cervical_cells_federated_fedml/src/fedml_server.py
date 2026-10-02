"""
FedML Server Aggregator & Centralized Holdout Evaluation Engine for Cervical Cytology.
Coordinates FedAvg, FedProx, and FedAdam parameter aggregation across hospital client silos.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from src.evaluator import evaluate_multiclass_metrics
from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    get_model_parameters,
    set_model_parameters,
)
from src.skorch_engine import evaluate_model_on_test_set_skorch
from src.wandb_tracker import WandbExperimentTracker


class FedMLCervicalServerAggregator:
    """
    FedML Server Aggregator coordinating global parameter updates
    and centralized holdout testing across federated communication rounds.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        test_df: pd.DataFrame,
        tracker: WandbExperimentTracker,
    ):
        self.cfg = cfg
        self.test_df = test_df
        self.tracker = tracker
        self.strategy_name = cfg.get("strategy", "FedAvg")
        self.framework = cfg.get("framework", "fastai").lower()

        self.global_model: nn.Module = build_cervical_model(
            arch_name=cfg.get("architecture", "convnext_small"),
            num_classes=int(cfg.get("num_classes", 7)),
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg.get("dropout_rate", 0.2)),
        )
        self.global_weights = get_model_parameters(self.global_model)

        # FedAdam server optimizer buffers
        self.adam_m: list[np.ndarray] | None = None
        self.adam_v: list[np.ndarray] | None = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.adam_eta = 0.01
        self.adam_tau = 1e-3

    def get_global_model_params(self) -> list[np.ndarray]:
        """Returns current global weights."""
        return self.global_weights

    def set_global_model_params(self, params: list[np.ndarray]) -> None:
        """Sets current global weights and updates PyTorch model."""
        self.global_weights = params
        set_model_parameters(self.global_model, params)

    def aggregate_fedavg(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """Sample-weighted parameter averaging across client silos."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.global_weights

        num_layers = len(self.global_weights)
        new_weights = [np.zeros_like(w, dtype=np.float64) for w in self.global_weights]

        for client_idx, c_weights in enumerate(client_weights):
            weight_factor = sample_counts[client_idx] / total_samples
            for layer_idx in range(num_layers):
                new_weights[layer_idx] += c_weights[layer_idx].astype(np.float64) * weight_factor

        return [w.astype(orig.dtype) for w, orig in zip(new_weights, self.global_weights, strict=False)]

    def aggregate_fedprox(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """Aggregates weights trained with FedProx proximal regularization."""
        return self.aggregate_fedavg(client_weights, sample_counts)

    def aggregate_fedadam(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """
        FedAdam adaptive server optimizer:
        Updates global model using first and second moment estimates of pseudo-gradients.
        """
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.global_weights

        num_layers = len(self.global_weights)
        # Average updated weights
        avg_weights = self.aggregate_fedavg(client_weights, sample_counts)

        # Server pseudo-gradient: Delta = avg_weights - w_global
        delta = [avg_weights[i] - self.global_weights[i] for i in range(num_layers)]

        # Initialize momentum buffers on round 1
        if self.adam_m is None:
            self.adam_m = [np.zeros_like(w) for w in self.global_weights]
            self.adam_v = [np.zeros_like(w) for w in self.global_weights]

        assert self.adam_v is not None

        new_weights = []
        for i in range(num_layers):
            # First moment: m_t = beta1 * m_{t-1} + (1 - beta1) * delta
            self.adam_m[i] = self.adam_beta1 * self.adam_m[i] + (1 - self.adam_beta1) * delta[i]
            # Second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * delta^2
            self.adam_v[i] = self.adam_beta2 * self.adam_v[i] + (1 - self.adam_beta2) * (delta[i] ** 2)

            # Global weight step
            step = self.adam_eta * self.adam_m[i] / (np.sqrt(self.adam_v[i]) + self.adam_tau)
            updated_layer = self.global_weights[i] + step
            new_weights.append(updated_layer.astype(self.global_weights[i].dtype))

        return new_weights

    def aggregate(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """Dispatches aggregation to the selected federated algorithm."""
        strat = self.strategy_name.lower()
        if "adam" in strat:
            updated = self.aggregate_fedadam(client_weights, sample_counts)
        elif "prox" in strat:
            updated = self.aggregate_fedprox(client_weights, sample_counts)
        else:
            updated = self.aggregate_fedavg(client_weights, sample_counts)

        self.set_global_model_params(updated)
        return updated

    def evaluate_global_model(
        self,
        round_idx: int,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluates the aggregated global model on the centralized holdout test set.
        Logs metrics to W&B tracker and returns (metrics, y_true, y_prob).
        """
        if self.framework == "skorch":
            y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
                model=self.global_model,
                test_df=self.test_df,
                cfg=self.cfg,
            )
        else:
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                model=self.global_model,
                test_df=self.test_df,
                cfg=self.cfg,
            )

        metrics = evaluate_multiclass_metrics(y_true, y_prob, num_classes=int(self.cfg.get("num_classes", 7)))
        metrics["test_loss"] = float(test_loss)

        # Log to W&B tracker
        log_payload = {f"eval/{k}": v for k, v in metrics.items()}
        log_payload["round"] = round_idx
        self.tracker.log_metrics(log_payload, step=round_idx)

        return metrics, y_true, y_prob

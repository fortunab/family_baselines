"""
Substra Federated Learning Engine for Cervical Cytology with VAE Preprocessing.
Conforms to Substra Asset Architecture (Opener, Algo, Strategy, and Central Platform).
Supports dual backends (fastai & skorch) with FedAvg, FedProx, and FedAdam aggregation.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch import nn

from src.evaluator import evaluate_multiclass_metrics
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
from src.wandb_tracker import WandbExperimentTracker


class SubstraCervicalOpener:
    """
    Substra Opener asset for multi-class cervical cytology images.
    Provides standardized data ingestion and parsing for Substra organization nodes.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def get_data(self, data: pd.DataFrame | str | Path) -> pd.DataFrame:
        """Loads DataFrame manifest for the Substra node."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        path = Path(data)
        if path.is_file() and path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.DataFrame()


class SubstraCervicalAlgo:
    """
    Substra Algorithm function executing local node training with VAE preprocessing.
    Supports both fastai and skorch backends.
    """

    def __init__(self, org_id: str, cfg: dict[str, Any]):
        self.org_id = org_id
        self.cfg = cfg
        self.framework = str(cfg.get("framework", "fastai")).lower()
        self.prep_mode = str(cfg.get("preprocessing", "vae")).lower()
        self.preprocessor = get_preprocessor(self.prep_mode, cfg)
        self.opener = SubstraCervicalOpener(image_size=int(cfg.get("image_size", 224)))

        arch_name = cfg.get("architecture", "convnext_small")
        num_classes = int(cfg.get("num_classes", 7))
        dropout = float(cfg.get("dropout_rate", 0.2))
        self.model: nn.Module = build_cervical_model(
            arch_name=arch_name,
            num_classes=num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=dropout,
        )

    def train(
        self,
        data: pd.DataFrame | str | Path,
        in_model_weights: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], float, int]:
        """Executes local training on Substra organization node."""
        client_df = self.opener.get_data(data)
        if in_model_weights is not None:
            set_model_parameters(self.model, in_model_weights)

        if self.framework == "skorch":
            updated_weights, loss, num_samples = local_train_skorch(
                model=self.model,
                client_df=client_df,
                cfg=self.cfg,
                global_model_weights=in_model_weights,
                preprocessor=self.preprocessor,
            )
        else:
            updated_weights, loss, num_samples = local_train_fastai(
                model=self.model,
                client_df=client_df,
                cfg=self.cfg,
                global_model_weights=in_model_weights,
                preprocessor=self.preprocessor,
            )

        return updated_weights, loss, num_samples


class SubstraFederatedStrategy:
    """Substra Aggregation Strategy supporting FedAvg, FedProx, and FedAdam."""

    def __init__(self, strategy_name: str = "FedAvg", cfg: dict[str, Any] | None = None):
        self.strategy_name = strategy_name
        self.cfg = cfg or {}
        self.adam_m: list[np.ndarray] | None = None
        self.adam_v: list[np.ndarray] | None = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.adam_eta = 1e-3
        self.adam_tau = 1e-3

    def aggregate(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
        global_weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Aggregates weights from organization nodes."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return global_weights

        num_layers = len(global_weights)
        new_weights = [np.zeros_like(w, dtype=np.float64) for w in global_weights]

        for client_idx, c_w in enumerate(client_weights):
            weight_factor = sample_counts[client_idx] / total_samples
            for l_idx in range(num_layers):
                new_weights[l_idx] += c_w[l_idx].astype(np.float64) * weight_factor

        avg_weights = [w.astype(orig.dtype) for w, orig in zip(new_weights, global_weights, strict=False)]

        strat = self.strategy_name.lower()
        if "adam" in strat:
            delta = [avg_weights[i] - global_weights[i] for i in range(num_layers)]
            if self.adam_m is None:
                self.adam_m = [np.zeros_like(w) for w in global_weights]
                self.adam_v = [np.zeros_like(w) for w in global_weights]

            updated_adam = []
            for i in range(num_layers):
                self.adam_m[i] = self.adam_beta1 * self.adam_m[i] + (1 - self.adam_beta1) * delta[i]
                self.adam_v[i] = self.adam_beta2 * self.adam_v[i] + (1 - self.adam_beta2) * (delta[i] ** 2)
                step = self.adam_eta * self.adam_m[i] / (np.sqrt(self.adam_v[i]) + self.adam_tau)
                layer = global_weights[i] + step
                updated_adam.append(layer.astype(global_weights[i].dtype))
            return updated_adam

        return avg_weights


class SubstraSimulatedPlatform:
    """
    Substra Platform orchestrating decentralized cross-silo collaborative learning,
    registering organization nodes, tracking federated rounds, and evaluating on holdout data.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        client_dfs: list[pd.DataFrame],
        test_df: pd.DataFrame,
        tracker: WandbExperimentTracker,
    ):
        self.cfg = cfg
        self.client_dfs = client_dfs
        self.test_df = test_df
        self.tracker = tracker

        self.num_classes = int(cfg.get("num_classes", 7))
        self.arch_name = cfg.get("architecture", "convnext_small")
        self.framework = str(cfg.get("framework", "fastai")).lower()
        self.prep_mode = str(cfg.get("preprocessing", "vae")).lower()
        self.strategy_name = str(cfg.get("strategy", "FedAvg"))

        self.global_model: nn.Module = build_cervical_model(
            arch_name=self.arch_name,
            num_classes=self.num_classes,
            pretrained=bool(cfg.get("pretrained", True)),
            dropout=float(cfg.get("dropout_rate", 0.2)),
        )
        self.global_weights = get_model_parameters(self.global_model)

        self.strategy = SubstraFederatedStrategy(self.strategy_name, cfg)
        self.preprocessor = get_preprocessor(self.prep_mode, cfg)

        # Register organization node algorithms
        self.org_algos = [
            SubstraCervicalAlgo(org_id=f"org_hospital_{i+1:02d}", cfg=cfg) for i in range(len(client_dfs))
        ]

    def run_training_round(self, round_idx: int) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Executes a single Substra federated communication round across organization nodes."""
        client_weights = []
        sample_counts = []

        for org_idx, (algo, c_df) in enumerate(zip(self.org_algos, self.client_dfs, strict=False)):
            updated_w, loss, n_samples = algo.train(c_df, in_model_weights=self.global_weights)
            client_weights.append(updated_w)
            sample_counts.append(n_samples)
            print(f"  [Substra Org {org_idx + 1:02d}] Finished: samples={n_samples}, loss={loss:.4f}")

        # Server-side aggregation
        self.global_weights = self.strategy.aggregate(client_weights, sample_counts, self.global_weights)
        set_model_parameters(self.global_model, self.global_weights)

        # Centralized holdout evaluation
        if self.framework == "skorch":
            y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
                model=self.global_model,
                test_df=self.test_df,
                cfg=self.cfg,
                preprocessor=self.preprocessor,
            )
        else:
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                model=self.global_model,
                test_df=self.test_df,
                cfg=self.cfg,
                preprocessor=self.preprocessor,
            )

        metrics = evaluate_multiclass_metrics(y_true, y_prob, num_classes=self.num_classes)
        metrics["test_loss"] = float(test_loss)

        log_payload = {f"eval/{k}": v for k, v in metrics.items()}
        log_payload["round"] = round_idx
        self.tracker.log_metrics(log_payload, step=round_idx)

        print(
            f"  [Substra Round {round_idx:02d} Holdout Eval] "
            f"Loss: {metrics.get('test_loss', 0.0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0.0):.4f} | "
            f"Bal Acc: {metrics.get('balanced_accuracy', 0.0):.4f} | "
            f"F1 Macro: {metrics.get('f1_macro', 0.0):.4f} | "
            f"AUC: {metrics.get('roc_auc_macro', 0.0):.4f}"
        )

        return metrics, y_true, y_prob

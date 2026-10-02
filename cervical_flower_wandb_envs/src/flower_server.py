"""
Flower (flwr) Server Strategy Builder & Centralized Holdout Evaluation for Cervical Cytology.
Provides FedAvg, FedProx, and FedAdam aggregation strategies with W&B round telemetry.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr
import pandas as pd
import torch.nn as nn
from flwr.common import NDArrays, Scalar

from src.evaluator import evaluate_multiclass_metrics
from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    set_model_parameters,
)
from src.wandb_tracker import WandbExperimentTracker


def get_server_evaluate_fn(
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tracker: WandbExperimentTracker,
    global_eval_history: List[Dict[str, Any]],
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """
    Returns an evaluation callback that tests the global aggregated model
    on the centralized unseen holdout test set after each federated round.
    """
    arch_name = cfg.get("architecture", "convnext_small")
    num_classes = int(cfg.get("num_classes", 7))
    eval_model: nn.Module = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout_rate", 0.2)),
    )

    def evaluate_round(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round == 0:
            return None

        # Inject aggregated weights
        set_model_parameters(eval_model, parameters)

        # Global holdout test evaluation
        y_true, y_prob, test_loss = evaluate_model_on_test_set(
            model=eval_model,
            test_df=test_df,
            cfg=cfg,
        )
        metrics = evaluate_multiclass_metrics(y_true, y_prob, loss=test_loss)

        # Stream to Weights & Biases
        log_payload = {
            "global_accuracy": metrics["accuracy"],
            "global_balanced_acc": metrics["balanced_accuracy"],
            "global_macro_f1": metrics["macro_f1"],
            "global_weighted_f1": metrics["weighted_f1"],
            "global_macro_roc_auc": metrics["macro_roc_auc"],
            "global_loss": metrics["loss"],
        }
        tracker.log_round_metrics(round_num=server_round, metrics=log_payload)
        global_eval_history.append({"round": server_round, **metrics})

        print(
            f"[Flower Server Eval R{server_round}] Holdout Acc: {metrics['accuracy'] * 100:.2f}% | "
            f"BalAcc: {metrics['balanced_accuracy'] * 100:.2f}% | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"ROC-AUC: {metrics['macro_roc_auc']:.4f} | "
            f"Loss: {test_loss:.4f}"
        )

        return float(test_loss), {
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }

    return evaluate_round


def create_flower_strategy(
    cfg: Dict[str, Any],
    initial_parameters: Any,
    evaluate_fn: Optional[Callable],
) -> flwr.server.strategy.Strategy:
    """Builds and configures the designated Flower federated aggregation strategy."""
    strategy_name = cfg.get("strategy", "FedAvg")
    fraction_fit = float(cfg.get("fraction_fit", 1.0))
    min_fit_clients = int(cfg.get("num_clients", 5))
    min_available_clients = int(cfg.get("num_clients", 5))

    if strategy_name == "FedProx":
        proximal_mu = float(cfg.get("proximal_mu", 1.0))
        print(f"[Federated-Server] Instantiating FedProx Strategy (mu={proximal_mu})...")
        strategy = flwr.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            proximal_mu=proximal_mu,
        )
    elif strategy_name == "FedAdam":
        print("[Federated-Server] Instantiating FedAdam Strategy...")
        strategy = flwr.server.strategy.FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            eta=0.01,
            beta_1=0.9,
            beta_2=0.99,
        )
    else:
        print("[Federated-Server] Instantiating FedAvg Standard Strategy...")
        strategy = flwr.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )

    return strategy

"""
Flower Server Strategies & Centralized Holdout Evaluation Callbacks for skorch FL.
Supports FedAvg, FedProx (heterogeneous data drift), and FedAdam adaptive aggregation.
"""

from collections.abc import Callable
from typing import Any

import flwr
import pandas as pd
from flwr.common import NDArrays, Scalar
from torch import nn

from src.evaluator import evaluate_multiclass_metrics
from src.skorch_engine import (
    build_cervical_model,
    evaluate_skorch_on_test_set,
    set_model_parameters,
)
from src.wandb_tracker import WandbExperimentTracker


def get_server_evaluate_fn(
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    tracker: WandbExperimentTracker,
    global_eval_history: list[dict[str, Any]],
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None]:
    """Generates server evaluation callback executed after each federated aggregation round."""
    arch_name = cfg.get("architecture", "convnext_small")
    num_classes = int(cfg.get("num_classes", 7))
    eval_model: nn.Module = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=False,
    )

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]] | None:
        if len(test_df) == 0:
            return None

        # Synchronize global aggregated weights into server evaluation model
        set_model_parameters(eval_model, parameters)

        y_true, y_prob, loss = evaluate_skorch_on_test_set(
            model=eval_model,
            test_df=test_df,
            cfg=cfg,
        )

        metrics = evaluate_multiclass_metrics(y_true, y_prob)
        metrics["loss"] = float(loss)
        metrics["round"] = int(server_round)
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_prob.argmax(axis=1) if len(y_prob) > 0 else []
        metrics["y_prob"] = y_prob

        global_eval_history.append(metrics)
        tracker.log_round_metrics(round_num=server_round, metrics=metrics)

        print(
            f"[Flower Server Eval R{server_round}] Holdout Acc: {metrics['accuracy'] * 100:.2f}% | "
            f"BalAcc: {metrics['balanced_accuracy'] * 100:.2f}% | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"ROC-AUC: {metrics['macro_roc_auc']:.4f} | "
            f"Loss: {loss:.4f}"
        )

        return (
            float(loss),
            {
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            },
        )

    return evaluate


def create_flower_strategy(
    cfg: dict[str, Any],
    initial_parameters: flwr.common.Parameters,
    evaluate_fn: Callable | None,
) -> flwr.server.strategy.Strategy:
    """Instantiates requested Flower aggregation strategy (FedAvg, FedProx, FedAdam)."""
    strat_name = cfg.get("strategy", "FedAvg").strip()
    fraction_fit = float(cfg.get("fraction_fit", 1.0))
    min_fit_clients = max(1, int(cfg.get("num_clients", 5) * fraction_fit))

    if strat_name.lower() == "fedprox":
        proximal_mu = float(cfg.get("proximal_mu", 1.0))
        print(f"[Federated-Server] Instantiating FedProx Strategy (mu = {proximal_mu})...")
        return flwr.server.strategy.FedProx(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=int(cfg.get("num_clients", 5)),
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            proximal_mu=proximal_mu,
        )

    elif strat_name.lower() == "fedadam":
        eta = float(cfg.get("eta", 1e-1))
        print(f"[Federated-Server] Instantiating FedAdam Adaptive Strategy (eta = {eta})...")
        return flwr.server.strategy.FedAdam(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=int(cfg.get("num_clients", 5)),
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            eta=eta,
        )

    else:
        print("[Federated-Server] Instantiating FedAvg Standard Strategy...")
        return flwr.server.strategy.FedAvg(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=int(cfg.get("num_clients", 5)),
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )

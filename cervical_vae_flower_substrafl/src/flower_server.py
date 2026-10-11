"""
Flower (flwr) Server Aggregation and Simulation Coordinator for Cervical Cytology.
Supports FedAvg, FedProx, FedAdam strategies, and centralized holdout evaluation.
"""

from collections.abc import Callable
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import Metrics, NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from torch import nn

from src.evaluator import evaluate_multiclass_metrics
from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    set_model_parameters,
)
from src.flower_client import FlowerCervicalClient
from src.preprocessing import get_preprocessor
from src.skorch_engine import evaluate_model_on_test_set_skorch
from src.wandb_tracker import WandbExperimentTracker


def get_evaluate_fn(
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    tracker: WandbExperimentTracker,
) -> Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]]]:
    """Returns a centralized evaluation callback for Flower server."""
    num_classes = int(cfg.get("num_classes", 7))
    arch_name = cfg.get("architecture", "convnext_small")
    framework = cfg.get("framework", "fastai").lower()
    prep_mode = cfg.get("preprocessing", "vae").lower()

    eval_model: nn.Module = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout_rate", 0.2)),
    )
    preprocessor = get_preprocessor(prep_mode, cfg)

    def evaluate(
        server_round: int,
        parameters: NDArrays | Parameters,
        config: dict[str, Scalar],
    ) -> tuple[float, dict[str, Scalar]]:
        if isinstance(parameters, Parameters):
            weights = parameters_to_ndarrays(parameters)
        else:
            weights = parameters
        set_model_parameters(eval_model, weights)

        if framework == "skorch":
            y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
                model=eval_model,
                test_df=test_df,
                cfg=cfg,
                preprocessor=preprocessor,
            )
        else:
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                model=eval_model,
                test_df=test_df,
                cfg=cfg,
                preprocessor=preprocessor,
            )

        metrics = evaluate_multiclass_metrics(y_true, y_prob, num_classes=num_classes)
        metrics["test_loss"] = float(test_loss)

        log_payload = {f"eval/{k}": v for k, v in metrics.items()}
        log_payload["round"] = server_round
        tracker.log_metrics(log_payload, step=server_round)

        print(
            f"  [Flower Round {server_round:02d} Holdout Eval] "
            f"Loss: {metrics.get('test_loss', 0.0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0.0):.4f} | "
            f"Bal Acc: {metrics.get('balanced_accuracy', 0.0):.4f} | "
            f"F1 Macro: {metrics.get('f1_macro', 0.0):.4f} | "
            f"AUC: {metrics.get('roc_auc_macro', 0.0):.4f}"
        )

        scalar_metrics: dict[str, Scalar] = {k: float(v) for k, v in metrics.items()}
        return float(test_loss), scalar_metrics

    return evaluate


def weighted_metrics_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics across clients weighted by sample count."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated: dict[str, float] = {}
    for num_examples, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + v * (num_examples / total_examples)

    return aggregated


def create_flower_strategy(
    strategy_name: str,
    initial_parameters: Parameters,
    evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]]],
    cfg: dict[str, Any],
) -> fl.server.strategy.Strategy:
    """Instantiates Flower federated aggregation strategy."""
    strat = strategy_name.lower()
    num_clients = int(cfg.get("num_clients", 5))

    if "prox" in strat:
        mu = float(cfg.get("proximal_mu", 1.0))
        return fl.server.strategy.FedProx(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda r: {"server_round": r},
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_metrics_average,
            proximal_mu=mu,
        )
    elif "adam" in strat:
        return fl.server.strategy.FedAdam(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_metrics_average,
            eta=1e-3,
            eta_l=float(cfg.get("local_lr", 0.0003)),
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
        )
    else:  # Default FedAvg
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda r: {"server_round": r},
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=weighted_metrics_average,
        )


def run_flower_simulation(
    client_dfs: list[pd.DataFrame],
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
    tracker: WandbExperimentTracker,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Runs a complete Flower federated learning simulation across clients."""
    num_classes = int(cfg.get("num_classes", 7))
    arch_name = cfg.get("architecture", "convnext_small")
    num_rounds = int(cfg.get("num_rounds", 5))
    strategy_name = cfg.get("strategy", "FedAvg")

    base_model = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout_rate", 0.2)),
    )
    initial_weights = [param.detach().cpu().numpy() for param in base_model.parameters()]
    initial_parameters = ndarrays_to_parameters(initial_weights)

    evaluate_fn = get_evaluate_fn(test_df=test_df, cfg=cfg, tracker=tracker)

    strategy = create_flower_strategy(
        strategy_name=strategy_name,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        cfg=cfg,
    )

    client_pool = [FlowerCervicalClient(client_id=i, client_df=df, cfg=cfg) for i, df in enumerate(client_dfs)]

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid) % len(client_pool)
        return client_pool[idx].to_client()

    print("\n" + "=" * 80)
    print(f"  COMMENCING FLOWER SIMULATION ({cfg.get('framework', 'fastai').upper()} - {strategy_name})")
    print("=" * 80)

    curr_weights = initial_weights
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(client_dfs),
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
        final_weights = client_pool[0].get_parameters()
    except (ImportError, ModuleNotFoundError, Exception) as err:
        print(
            f"[Flower-Simulation] Ray simulation unavailable or errored ({err}); executing in-process simulation loop..."
        )
        for r in range(1, num_rounds + 1):
            print(f"\n--- [Flower Round {r:02d}/{num_rounds:02d}] ---")
            results = []
            for client in client_pool:
                w, num_samples, _ = client.fit(curr_weights, {"proximal_mu": float(cfg.get("proximal_mu", 1.0))})
                results.append((w, num_samples))

            total_samples = sum(n for _, n in results)
            accum_weights = [np.zeros_like(p) for p in curr_weights]
            for w_arrs, n_samp in results:
                frac = n_samp / max(1, total_samples)
                for i, arr in enumerate(w_arrs):
                    accum_weights[i] += frac * arr
            curr_weights = accum_weights

            eval_res = evaluate_fn(r, curr_weights, {})
            if eval_res is not None:
                loss_val, eval_dict = eval_res
                print(
                    f"  Holdout Loss: {loss_val:.4f} | Acc: {eval_dict.get('accuracy', 0.0):.4f} | F1: {eval_dict.get('f1_macro', 0.0):.4f}"
                )
        final_weights = curr_weights
    set_model_parameters(base_model, final_weights)
    preprocessor = get_preprocessor(cfg.get("preprocessing", "vae"), cfg)

    if cfg.get("framework", "fastai") == "skorch":
        y_true, y_prob, test_loss = evaluate_model_on_test_set_skorch(
            model=base_model,
            test_df=test_df,
            cfg=cfg,
            preprocessor=preprocessor,
        )
    else:
        y_true, y_prob, test_loss = evaluate_model_on_test_set(
            model=base_model,
            test_df=test_df,
            cfg=cfg,
            preprocessor=preprocessor,
        )

    final_metrics = evaluate_multiclass_metrics(y_true, y_prob, num_classes=num_classes)
    final_metrics["test_loss"] = float(test_loss)

    return final_metrics, y_true, y_prob

"""
Main Orchestration Script for Cervical Cytology Federated Learning with Flower (flwr).
Executes simulated federated learning across screening clinic clients,
evaluating on a centralized 15% holdout test set with Weights & Biases telemetry.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import flwr
import numpy as np
import torch
import torch.nn as nn

from src.dataset import prepare_cervical_partitions
from src.evaluator import (
    plot_confusion_matrix,
    plot_multiclass_roc_curves,
    print_evaluation_report,
)
from src.fastai_engine import build_cervical_model, get_model_parameters
from src.flower_client import make_client_fn
from src.flower_server import create_flower_strategy, get_server_evaluate_fn
from src.toml_config import load_toml_config, merge_cli_args, print_config_summary
from src.wandb_tracker import WandbExperimentTracker


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flower Federated Learning for Cervical Cytology (Herlev Pap Smear)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML experiment configuration file",
    )
    parser.add_argument("--arch", type=str, default=None, help="Vision backbone architecture")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["FedAvg", "FedProx", "FedAdam"],
        help="Federated aggregation strategy",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=None, help="Federated communication rounds"
    )
    parser.add_argument("--num_clients", type=int, default=None, help="Number of clinic clients")
    parser.add_argument("--local_epochs", type=int, default=None, help="Local epochs per clinic")
    parser.add_argument("--lr", type=float, default=None, help="Local client learning rate")
    parser.add_argument(
        "--proximal_mu", type=float, default=None, help="FedProx proximal regularizer parameter"
    )
    parser.add_argument(
        "--non_iid", action="store_true", help="Activate Non-IID Dirichlet distribution"
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=None, help="Dirichlet concentration parameter"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random reproducibility seed")
    parser.add_argument(
        "--subsample", type=int, default=None, help="Subsample dataset for fast dry-run"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Load TOML Configuration & Merge CLI Overrides
    raw_cfg = load_toml_config(args.config)
    cfg = merge_cli_args(raw_cfg, args)

    active_seed = int(cfg.get("seed", 42))
    set_seed(active_seed)
    print_config_summary(cfg)

    # 2. Prepare Cervical Cytology Partitions & 15% Holdout Test Set
    client_dfs, test_df = prepare_cervical_partitions(cfg)

    # 3. Initialize MLOps & Experiment Tracking
    tracker = WandbExperimentTracker(cfg)

    # 4. Initialize Global Model & Parameters
    strategy_name = cfg.get("strategy", "FedAvg")
    arch_name = cfg.get("architecture", "convnext_small")
    num_rounds = int(cfg.get("num_rounds", 10))
    num_clients = len(client_dfs)
    num_classes = int(cfg.get("num_classes", 7))

    init_model: nn.Module = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout_rate", 0.2)),
    )
    initial_weights = get_model_parameters(init_model)
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_weights)

    global_eval_history: List[Dict[str, Any]] = []

    server_eval_fn = get_server_evaluate_fn(
        test_df=test_df,
        cfg=cfg,
        tracker=tracker,
        global_eval_history=global_eval_history,
    )

    strategy = create_flower_strategy(
        cfg=cfg,
        initial_parameters=initial_parameters,
        evaluate_fn=server_eval_fn,
    )
    client_fn = make_client_fn(client_dfs=client_dfs, cfg=cfg)

    # 5. Start Flower Federated Simulation
    print(
        f"\n[Flower-Engine] Starting Cervical Cytology Federated Simulation ({num_rounds} rounds, {num_clients} clinics, {strategy_name})..."
    )
    server_config = flwr.server.ServerConfig(num_rounds=num_rounds)

    # Bound concurrency to at most 2 simultaneous virtual clients to prevent memory exhaustion
    total_cpus = os.cpu_count() or 4
    client_cpus = max(1, total_cpus // 2)

    flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=server_config,
        strategy=strategy,
        client_resources={"num_cpus": client_cpus, "num_gpus": 0.0},
    )

    # 6. Final Evaluation Reporting and Artifact Generation
    print("\n[Flower-Engine] Federated simulation completed! Generating final global evaluation...")
    if global_eval_history:
        latest_eval = global_eval_history[-1]
        print_evaluation_report(
            latest_eval,
            f"Global Cervical Model ({strategy_name} - {arch_name})",
            seed=active_seed,
        )

        output_dir = Path(cfg.get("output_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("save_plots", True) and "y_true" in latest_eval and "y_pred" in latest_eval:
            cm_path = output_dir / f"cervical_cm_{strategy_name.lower()}_{arch_name}.png"
            roc_path = output_dir / f"cervical_roc_{strategy_name.lower()}_{arch_name}.png"

            plot_confusion_matrix(
                y_true=latest_eval["y_true"],
                y_pred=latest_eval["y_pred"],
                output_path=str(cm_path),
                title=f"Cervical Cytology Confusion Matrix ({strategy_name} - {arch_name})",
            )
            print(f"[Artifacts] Saved Confusion Matrix to: {cm_path}")
            tracker.log_artifact_file(str(cm_path), f"cm_{strategy_name.lower()}_{arch_name}")

            plot_multiclass_roc_curves(
                y_true=latest_eval["y_true"],
                y_prob=latest_eval["y_prob"],
                output_path=str(roc_path),
                title=f"Cervical Cytology Multi-Class ROC Curves ({strategy_name} - {arch_name})",
            )
            print(f"[Artifacts] Saved ROC Curves to: {roc_path}")
            tracker.log_artifact_file(str(roc_path), f"roc_{strategy_name.lower()}_{arch_name}")

        tracker.save_final_summary(latest_eval)

    tracker.finish()
    print(
        f"\n[Completed] Cervical Cytology Federated Flower experiment for '{arch_name}' ({strategy_name}) finished successfully!"
    )


if __name__ == "__main__":
    main()

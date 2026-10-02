"""
Main Orchestration Script for Cervical Cytology Federated Learning with Flower & skorch.
Executes simulated federated learning across screening clinic clients using Scikit-Learn API,
evaluating on a centralized 15% holdout test set with Weights & Biases telemetry.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import flwr
import numpy as np
import torch
from torch import nn

from src.dataset import prepare_cervical_partitions
from src.evaluator import (
    plot_confusion_matrix,
    plot_multiclass_roc_curves,
    print_evaluation_report,
)
from src.flower_client import make_client_fn
from src.flower_server import create_flower_strategy, get_server_evaluate_fn
from src.skorch_engine import build_cervical_model, get_model_parameters
from src.toml_config import load_toml_config, merge_cli_args, print_config_summary
from src.wandb_tracker import WandbExperimentTracker

# Force line buffering for live real-time console and log flushing
sys.stdout.reconfigure(line_buffering=True)


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments and configuration overrides."""
    parser = argparse.ArgumentParser(description="Cervical Cytology Federated Learning with Flower (flwr) & skorch")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="FL Strategy (FedAvg, FedProx, FedAdam)",
    )
    parser.add_argument("--architecture", type=str, default=None, help="Vision backbone architecture")
    parser.add_argument("--num_rounds", type=int, default=None, help="Communication rounds")
    parser.add_argument("--num_clients", type=int, default=None, help="Participating hospital clients")
    parser.add_argument("--local_epochs", type=int, default=None, help="Local skorch epochs per client")
    parser.add_argument("--local_lr", type=float, default=None, help="Local learning rate")
    parser.add_argument("--proximal_mu", type=float, default=None, help="FedProx proximal parameter mu")
    parser.add_argument(
        "--non_iid",
        action="store_true",
        default=None,
        help="Enable Dirichlet non-IID skew",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Dirichlet concentration parameter",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random reproducibility seed")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast dry-run")
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

    global_eval_history: list[dict[str, Any]] = []

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

    # 5. Start Flower Federated Simulation with Concurrency Bounding
    print(
        f"\n[Flower-Engine] Starting Cervical Cytology Federated skorch Simulation ({num_rounds} rounds, {num_clients} clinics, {strategy_name})..."
    )
    server_config = flwr.server.ServerConfig(num_rounds=num_rounds)

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
            f"Global Cervical Model ({strategy_name} - {arch_name} via skorch)",
            seed=active_seed,
        )

        output_dir = Path(cfg.get("output_dir", "results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.get("save_plots", True) and "y_true" in latest_eval and "y_pred" in latest_eval:
            cm_path = output_dir / f"cervical_cm_{strategy_name.lower()}_{arch_name}.png"
            roc_path = output_dir / f"cervical_roc_{strategy_name.lower()}_{arch_name}.png"

            plot_confusion_matrix(
                latest_eval["y_true"],
                latest_eval["y_pred"],
                cm_path,
                title=f"Global Cervical {strategy_name} ({arch_name} + skorch) Confusion Matrix",
            )
            plot_multiclass_roc_curves(
                latest_eval["y_true"],
                latest_eval["y_prob"],
                roc_path,
                title=f"Global Cervical {strategy_name} ({arch_name} + skorch) Multi-Class ROC",
            )

            tracker.log_artifact(cm_path, "confusion_matrix")
            tracker.log_artifact(roc_path, "roc_curves")

        tracker.save_final_summary(output_dir, latest_eval)

    tracker.finish()
    print(
        f"[Completed] Cervical Cytology Federated skorch experiment for '{arch_name}' ({strategy_name}) finished successfully!"
    )


if __name__ == "__main__":
    main()

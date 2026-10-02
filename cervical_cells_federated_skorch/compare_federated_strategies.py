"""
Comparative Benchmark Script for Flower Federated Strategies on Cervical Cytology with skorch.
Directly compares FedAvg vs FedProx vs FedAdam on identical clinic data partitions.
"""

import os
import sys
from pathlib import Path
from typing import Any

import flwr
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn

from src.dataset import prepare_cervical_partitions
from src.flower_client import make_client_fn
from src.flower_server import create_flower_strategy, get_server_evaluate_fn
from src.skorch_engine import build_cervical_model, get_model_parameters
from src.toml_config import load_toml_config
from src.wandb_tracker import WandbExperimentTracker

# Force line buffering
sys.stdout.reconfigure(line_buffering=True)


def run_strategy_benchmark(
    strategy_name: str,
    base_cfg: dict[str, Any],
    client_dfs: list[pd.DataFrame],
    test_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs a single federated learning strategy benchmark with skorch."""
    cfg = base_cfg.copy()
    cfg["strategy"] = strategy_name
    cfg["run_name"] = f"benchmark_skorch_{strategy_name.lower()}"

    arch_name = cfg.get("architecture", "convnext_small")
    num_rounds = int(cfg.get("num_rounds", 3))
    num_clients = len(client_dfs)
    num_classes = int(cfg.get("num_classes", 7))

    tracker = WandbExperimentTracker(cfg)
    global_eval_history: list[dict[str, Any]] = []

    init_model: nn.Module = build_cervical_model(
        arch_name=arch_name,
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
    )
    initial_weights = get_model_parameters(init_model)
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_weights)

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

    tracker.finish()
    latest_metrics = global_eval_history[-1] if global_eval_history else {}
    return global_eval_history, latest_metrics


def main():
    print("=" * 80)
    print("   CERVICAL CYTOLOGY FEDERATED STRATEGY COMPARISON (FedAvg vs FedProx vs FedAdam - skorch)")
    print("=" * 80)

    raw_cfg = load_toml_config("configs/default.toml")
    flat_cfg = raw_cfg.get("general", {})
    flat_cfg.update(raw_cfg.get("dataset", {}))
    flat_cfg.update(raw_cfg.get("model", {}))
    flat_cfg.update(raw_cfg.get("federated", {}))

    flat_cfg["num_rounds"] = 3
    flat_cfg["num_clients"] = 3
    flat_cfg["local_epochs"] = 1
    flat_cfg["subsample"] = 60
    flat_cfg["wandb_enabled"] = False

    client_dfs, test_df = prepare_cervical_partitions(flat_cfg)

    strategies = ["FedAvg", "FedProx", "FedAdam"]
    results = {}
    summary_rows = []

    for strat in strategies:
        print(f"\n>>> Running Cervical Cytology skorch Benchmark for: {strat} <<<")
        history, final_metrics = run_strategy_benchmark(
            strategy_name=strat,
            base_cfg=flat_cfg,
            client_dfs=client_dfs,
            test_df=test_df,
        )
        results[strat] = history
        summary_rows.append(
            {
                "Strategy": strat,
                "Accuracy": f"{final_metrics.get('accuracy', 0.0) * 100:.2f}%",
                "Balanced_Accuracy": f"{final_metrics.get('balanced_accuracy', 0.0) * 100:.2f}%",
                "Macro_F1": f"{final_metrics.get('macro_f1', 0.0):.4f}",
                "Macro_ROC_AUC": f"{final_metrics.get('macro_roc_auc', 0.0):.4f}",
                "Final_Loss": f"{final_metrics.get('loss', 0.0):.4f}",
            }
        )

    # Save CSV Summary
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "cervical_skorch_strategy_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Benchmark] Saved summary table to: {csv_path}")
    print(summary_df.to_string(index=False))

    # Plot 4-Panel Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"FedAvg": "#1f77b4", "FedProx": "#2ca02c", "FedAdam": "#d62728"}
    markers = {"FedAvg": "o", "FedProx": "s", "FedAdam": "^"}

    for strat, hist in results.items():
        rounds = [r["round"] for r in hist]
        accs = [r["accuracy"] * 100 for r in hist]
        bal_accs = [r["balanced_accuracy"] * 100 for r in hist]
        f1s = [r["macro_f1"] for r in hist]
        losses = [r["loss"] for r in hist]

        c = colors.get(strat, "black")
        m = markers.get(strat, "o")

        axes[0, 0].plot(rounds, accs, marker=m, color=c, lw=2, label=strat)
        axes[0, 1].plot(rounds, bal_accs, marker=m, color=c, lw=2, label=strat)
        axes[1, 0].plot(rounds, f1s, marker=m, color=c, lw=2, label=strat)
        axes[1, 1].plot(rounds, losses, marker=m, color=c, lw=2, label=strat)

    axes[0, 0].set_title("Centralized Holdout Accuracy (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].set_title("Balanced Accuracy (%)", fontweight="bold")
    axes[0, 1].set_ylabel("Balanced Accuracy (%)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].set_title("Macro F1-Score", fontweight="bold")
    axes[1, 0].set_xlabel("Communication Round")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].set_title("Centralized Evaluation Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Communication Round")
    axes[1, 1].set_ylabel("CrossEntropy Loss")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(
        "Cervical Cytology Federated Strategy Convergence (skorch Engine)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plot_path = output_dir / "cervical_skorch_strategy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[Benchmark] Saved 4-Panel comparison plot to: {plot_path}")


if __name__ == "__main__":
    main()

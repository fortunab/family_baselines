"""
Comparative Benchmark Script for Flower Federated Strategies on Cervical Cytology.
Directly compares FedAvg vs FedProx vs FedAdam on identical clinic data partitions.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn

from src.dataset import prepare_cervical_partitions
from src.fastai_engine import build_cervical_model, get_model_parameters
from src.flower_client import make_client_fn
from src.flower_server import create_flower_strategy, get_server_evaluate_fn
from src.toml_config import load_toml_config
from src.wandb_tracker import WandbExperimentTracker


def run_strategy_benchmark(
    strategy_name: str,
    base_cfg: Dict[str, Any],
    client_dfs: List[pd.DataFrame],
    test_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs a single federated experiment with the designated Flower strategy."""
    cfg = base_cfg.copy()
    cfg["strategy"] = strategy_name
    cfg["run_name"] = f"benchmark_cervical_{strategy_name.lower()}"
    cfg["wandb_enabled"] = False

    if strategy_name == "FedProx":
        cfg["proximal_mu"] = float(cfg.get("proximal_mu", 1.0))

    tracker = WandbExperimentTracker(cfg)
    global_eval_history: List[Dict[str, Any]] = []

    arch_name = cfg.get("architecture", "convnext_small")
    num_classes = int(cfg.get("num_classes", 7))
    num_rounds = int(cfg.get("num_rounds", 4))
    num_clients = len(client_dfs)

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
    print("   CERVICAL CYTOLOGY FEDERATED STRATEGY COMPARISON (FedAvg vs FedProx vs FedAdam)")
    print("=" * 80)

    cfg_path = "configs/fedprox_non_iid.toml"
    raw_cfg = load_toml_config(cfg_path)
    flat_cfg = {
        **raw_cfg.get("general", {}),
        **raw_cfg.get("dataset", {}),
        **raw_cfg.get("model", {}),
        **raw_cfg.get("federated", {}),
    }
    flat_cfg["num_rounds"] = 4
    flat_cfg["num_clients"] = 3
    flat_cfg["local_epochs"] = 1
    flat_cfg["subsample"] = 90
    flat_cfg["wandb_enabled"] = False

    client_dfs, test_df = prepare_cervical_partitions(flat_cfg)

    strategies = ["FedAvg", "FedProx", "FedAdam"]
    results = {}
    summary_rows = []

    for strat in strategies:
        print(f"\n>>> Running Cervical Cytology Benchmark for: {strat} <<<")
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
    csv_path = output_dir / "cervical_strategy_summary.csv"
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

        axes[0, 0].plot(rounds, accs, label=strat, color=colors[strat], marker=markers[strat], lw=2)
        axes[0, 1].plot(
            rounds, bal_accs, label=strat, color=colors[strat], marker=markers[strat], lw=2
        )
        axes[1, 0].plot(rounds, f1s, label=strat, color=colors[strat], marker=markers[strat], lw=2)
        axes[1, 1].plot(
            rounds, losses, label=strat, color=colors[strat], marker=markers[strat], lw=2
        )

    axes[0, 0].set_title("Global Holdout Accuracy (%)", fontweight="bold")
    axes[0, 0].set_xlabel("Communication Round")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend()

    axes[0, 1].set_title("Global Balanced Accuracy (%)", fontweight="bold")
    axes[0, 1].set_xlabel("Communication Round")
    axes[0, 1].set_ylabel("Balanced Accuracy (%)")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend()

    axes[1, 0].set_title("Global Macro F1-Score", fontweight="bold")
    axes[1, 0].set_xlabel("Communication Round")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend()

    axes[1, 1].set_title("Global Test Cross-Entropy Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Communication Round")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend()

    plt.suptitle(
        "Cervical Cytology FL: Strategy Convergence Comparison", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    plot_path = output_dir / "cervical_strategy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Benchmark] Saved comparison plots to: {plot_path}")


if __name__ == "__main__":
    main()

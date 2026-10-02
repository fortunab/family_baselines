"""
Systematic Benchmark Runner: FedML Strategies & Dual Framework Comparison.
Compares FedAvg, FedProx, and FedAdam across fastai and skorch engines on Herlev cervical cytology.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add current folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import prepare_fedml_partitions
from src.fedml_client import FedMLClientCervicalTrainer
from src.fedml_server import FedMLCervicalServerAggregator
from src.seed_selector import find_best_partition_seed
from src.toml_config import load_toml_config
from src.wandb_tracker import WandbExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedML Strategy & Framework Comparison")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Base config")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per benchmark configuration")
    parser.add_argument("--clients", type=int, default=3, help="Number of hospital clients")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--subsample", type=int, default=20, help="Subsample limit for comparison")
    parser.add_argument("--no_wandb", action="store_true", default=True, help="Disable W&B")
    return parser.parse_args()


def run_single_experiment(
    framework: str,
    strategy: str,
    base_cfg: dict,
    client_dfs: list,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    cfg = base_cfg.copy()
    cfg["framework"] = framework
    cfg["strategy"] = strategy
    cfg["num_rounds"] = args.rounds
    cfg["num_clients"] = args.clients
    cfg["local_epochs"] = args.epochs
    cfg["wandb_enabled"] = not args.no_wandb

    tracker = WandbExperimentTracker(cfg)
    server = FedMLCervicalServerAggregator(cfg=cfg, test_df=test_df, tracker=tracker)
    clients = [FedMLClientCervicalTrainer(client_id=i, cfg=cfg) for i in range(len(client_dfs))]

    best_f1 = 0.0
    final_metrics = {}

    for r_idx in range(1, args.rounds + 1):
        global_weights = server.get_global_model_params()
        c_weights, c_samples = [], []
        for client, c_df in zip(clients, client_dfs, strict=False):
            w, _, n = client.train(train_data=c_df, global_model_weights=global_weights)
            c_weights.append(w)
            c_samples.append(n)

        server.aggregate(c_weights, c_samples)
        metrics, _, _ = server.evaluate_global_model(round_idx=r_idx)
        final_metrics = metrics
        if metrics.get("f1_macro", 0.0) > best_f1:
            best_f1 = metrics["f1_macro"]

    tracker.finish()
    return {
        "framework": framework,
        "strategy": strategy,
        "best_f1_macro": best_f1,
        "final_accuracy": final_metrics.get("accuracy", 0.0),
        "final_balanced_acc": final_metrics.get("balanced_accuracy", 0.0),
        "final_auc": final_metrics.get("roc_auc_macro", 0.0),
    }


def main() -> None:
    args = parse_args()
    base_cfg = load_toml_config(args.config)
    base_cfg["num_clients"] = args.clients

    print("=" * 80)
    print("  FEDML CERVICAL CYTOLOGY STRATEGY & FRAMEWORK BENCHMARK")
    print("=" * 80)

    # Multi-candidate Random Best Select seed optimization
    best_seed, client_dfs, test_df, _ = find_best_partition_seed(
        partition_fn=lambda s: prepare_fedml_partitions(base_cfg, seed=s),
        num_candidates=3,
        num_classes=int(base_cfg.get("num_classes", 7)),
    )
    base_cfg["seed"] = best_seed

    # Apply subsample for comparison
    if args.subsample > 0:
        client_dfs = [df.iloc[: args.subsample].copy() for df in client_dfs]
        test_df = test_df.iloc[: max(10, args.subsample * 2)].copy()

    experiments = [
        ("fastai", "FedAvg"),
        ("fastai", "FedProx"),
        ("fastai", "FedAdam"),
        ("skorch", "FedAvg"),
        ("skorch", "FedProx"),
        ("skorch", "FedAdam"),
    ]

    results = []
    for fw, strat in experiments:
        print(f"\n[*] Benchmarking {fw.upper()} with {strat}...")
        res = run_single_experiment(fw, strat, base_cfg, client_dfs, test_df, args)
        results.append(res)
        print(f"    --> F1-Macro: {res['best_f1_macro']:.4f} | Accuracy: {res['final_accuracy']:.4f}")

    df_results = pd.DataFrame(results)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fedml_strategy_comparison.csv"
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Bar chart
    plt.figure(figsize=(10, 6))
    labels = [f"{r['framework']}\n{r['strategy']}" for r in results]
    f1s = [r["best_f1_macro"] for r in results]
    accs = [r["final_accuracy"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width / 2, f1s, width, label="F1-Macro", color="#2b5c8f")
    plt.bar(x + width / 2, accs, width, label="Accuracy", color="#4ba3e3")

    plt.ylabel("Score", fontsize=12)
    plt.title("FedML Strategy & Framework Comparison on Cervical Cytology", fontsize=14, pad=15)
    plt.xticks(x, labels, fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plot_path = out_dir / "fedml_strategy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n[Artifacts] Saved comparison CSV to {csv_path} and plot to {plot_path}")


if __name__ == "__main__":
    main()

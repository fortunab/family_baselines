"""
Multi-Strategy & Dual-Framework Benchmark Comparison Script for Substra Cervical Cytology.
Compares FedAvg vs FedProx vs FedAdam across fastai and skorch engines,
evaluating non-IID resilience and convergence stability with Random Best Select.
"""

import argparse
import json
import os
from pathlib import Path

# OpenMP safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd

from src.dataset import prepare_cervical_partitions
from src.seed_selector import select_best_random_seed
from src.substra_orchestrator import SubstraComputePlanOrchestrator
from src.toml_config import load_toml_config
from src.wandb_tracker import WandbExperimentTracker


def parse_arguments() -> argparse.Namespace:
    """Parses benchmark comparison CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Substra Multi-Strategy & Dual-Framework Benchmark Comparison",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Base TOML configuration.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["FedAvg", "FedProx"],
        help="List of federated strategies to compare (e.g. FedAvg FedProx FedAdam).",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["fastai", "skorch"],
        help="Machine learning engines to evaluate (fastai, skorch).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds per benchmark run.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=70,
        help="Subsample size for fast benchmark comparison.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs systematic benchmark grid across strategies and frameworks."""
    args = parse_arguments()
    base_cfg = load_toml_config(Path(args.config))
    base_cfg["num_rounds"] = args.rounds
    base_cfg["subsample"] = args.subsample
    base_cfg["wandb_enabled"] = False  # Local evaluation table

    # 1. Random Best Select seed selection (No Seed 42)
    def partition_trial(cand_seed: int):
        return prepare_cervical_partitions(base_cfg, seed=cand_seed)

    best_seed, _ = select_best_random_seed(partition_trial, num_candidates=5, verbose=True)
    base_cfg["seed"] = best_seed
    base_cfg["active_seed"] = best_seed

    client_dfs, test_df = prepare_cervical_partitions(base_cfg, seed=best_seed)

    results = []

    print("\n" + "=" * 80)
    print("    SUBSTRA MULTI-STRATEGY & DUAL-FRAMEWORK BENCHMARK GRID")
    print(f"    Evaluating Strategies: {args.strategies} | Engines: {args.frameworks}")
    print(f"    Active Best Seed: {best_seed} (No Seed 42)")
    print("=" * 80 + "\n")

    for framework in args.frameworks:
        for strategy in args.strategies:
            print(f"\n>>> Running Substra Benchmark: Strategy={strategy} | Framework={framework.upper()} <<<")
            cfg = dict(base_cfg)
            cfg["strategy"] = strategy
            cfg["framework"] = framework
            cfg["run_name"] = f"benchmark_substra_{strategy.lower()}_{framework}"

            tracker = WandbExperimentTracker(cfg)
            orchestrator = SubstraComputePlanOrchestrator(
                cfg=cfg,
                client_dfs=client_dfs,
                test_df=test_df,
                tracker=tracker,
                active_seed=best_seed,
            )

            _, _, metrics = orchestrator.run_compute_plan()
            results.append(
                {
                    "Strategy": strategy,
                    "Framework": framework,
                    "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
                    "Balanced_Acc": f"{metrics['balanced_accuracy'] * 100:.2f}%",
                    "Macro_F1": f"{metrics['macro_f1']:.4f}",
                    "ROC_AUC": f"{metrics['macro_roc_auc']:.4f}",
                    "Loss": f"{metrics['loss']:.4f}",
                }
            )
            tracker.finish()

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("    SUBSTRA BENCHMARK RESULTS SUMMARY (Cervical Cytology 7-Class)")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80 + "\n")

    output_dir = Path(base_cfg.get("output_dir", "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "substra_framework_strategy_comparison.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Artifacts] Saved benchmark grid summary to: {summary_file}")


if __name__ == "__main__":
    main()

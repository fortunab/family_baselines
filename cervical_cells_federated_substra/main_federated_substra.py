"""
Main Execution Script for Cervical Cytology Federated Learning Substra Suite.
Supports dual-framework execution: fastai (1-Cycle policy) & skorch (Scikit-Learn API).
Features Substra Compute Plan DAG, dynamic high-entropy seeds with Random Best Select (No Seed 42),
and comprehensive clinical telemetry with Weights & Biases.
"""

import argparse
import os
from pathlib import Path

# OpenMP safety on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.dataset import prepare_cervical_partitions
from src.evaluator import (
    plot_cervical_confusion_matrix,
    plot_cervical_roc_curves,
    print_cervical_evaluation_report,
)
from src.seed_selector import (
    apply_random_seed,
    generate_dynamic_seed,
    select_best_random_seed,
)
from src.substra_orchestrator import SubstraComputePlanOrchestrator
from src.toml_config import (
    load_toml_config,
    merge_cli_arguments,
    print_substra_configuration_banner,
)
from src.wandb_tracker import WandbExperimentTracker


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for Substra federated execution."""
    parser = argparse.ArgumentParser(
        description="Substra Federated Cervical Cytology Dysplasia Grading (fastai + skorch)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML configuration profile.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["fastai", "skorch"],
        default=None,
        help="Machine learning engine to execute in Substra nodes ('fastai' or 'skorch').",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["FedAvg", "FedProx", "FedAdam"],
        default=None,
        help="Federated aggregation strategy.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Total federated aggregation rounds in compute plan.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of Substra hospital organization nodes.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="Local training epochs per Substra task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit seed (seed 42 is strictly rejected). Default performs Random Best Select.",
    )
    parser.add_argument(
        "--no_random_best_select",
        action="store_true",
        help="Disable multi-candidate Random Best Select and use single dynamic seed.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample dataset for fast dry-run verification.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases MLOps telemetry logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for Substra federated cervical cytology training."""
    args = parse_arguments()
    config_file = Path(args.config)
    cfg = load_toml_config(config_file)

    # CLI Overrides
    cli_overrides = {
        "framework": args.framework,
        "strategy": args.strategy,
        "num_rounds": args.num_rounds,
        "num_clients": args.num_clients,
        "local_epochs": args.local_epochs,
        "subsample": args.subsample,
    }
    if args.no_wandb:
        cli_overrides["wandb_enabled"] = False
    cfg = merge_cli_arguments(cfg, cli_overrides)

    # 1. Dynamic Seed Resolution & "Random Best Select" (No Seed 42)
    explicit_seed = args.seed if args.seed is not None else cfg.get("seed", None)
    if isinstance(explicit_seed, str) and explicit_seed.lower() in ("random", "random_best", "none"):
        explicit_seed = None

    if explicit_seed is not None and int(explicit_seed) != 42:
        active_seed = apply_random_seed(int(explicit_seed))
        print(f"[Seed-Manager] Using explicit high-entropy seed: {active_seed}")
        client_dfs, test_df = prepare_cervical_partitions(cfg, seed=active_seed)
    else:
        if explicit_seed == 42:
            print("[Seed-Manager] Seed 42 detected. Rejecting per benchmark instructions.")

        if args.no_random_best_select:
            active_seed = generate_dynamic_seed()
            apply_random_seed(active_seed)
            print(f"[Seed-Manager] Generated dynamic random seed: {active_seed} (No Seed 42)")
            client_dfs, test_df = prepare_cervical_partitions(cfg, seed=active_seed)
        else:
            # Execute "Random Best Select" multi-candidate optimizer
            def partition_trial(trial_seed: int):
                return prepare_cervical_partitions(cfg, seed=trial_seed)

            active_seed, _ = select_best_random_seed(partition_trial, num_candidates=5, verbose=True)
            client_dfs, test_df = prepare_cervical_partitions(cfg, seed=active_seed)

    cfg["active_seed"] = active_seed

    # Print Configuration Banner
    print_substra_configuration_banner(cfg, active_seed=active_seed)

    # 2. Initialize MLOps Telemetry
    tracker = WandbExperimentTracker(cfg)

    # 3. Initialize Substra Compute Plan DAG Orchestrator
    orchestrator = SubstraComputePlanOrchestrator(
        cfg=cfg,
        client_dfs=client_dfs,
        test_df=test_df,
        tracker=tracker,
        active_seed=active_seed,
    )

    # 4. Execute Compute Plan DAG
    y_true, y_prob, final_metrics = orchestrator.run_compute_plan()

    # 5. Diagnostic Reporting & Artifact Generation
    output_dir = Path(cfg.get("output_dir", "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_name = cfg.get("strategy", "FedAvg")
    arch_name = cfg.get("architecture", "convnext_small")
    framework_name = cfg.get("framework", "fastai")

    print_cervical_evaluation_report(
        metrics=final_metrics,
        y_true=y_true,
        y_prob=y_prob,
        strategy_name=strategy_name,
        arch_name=arch_name,
        seed=active_seed,
        framework=framework_name,
    )

    cm_path = output_dir / f"cervical_cm_substra_{strategy_name.lower()}_{framework_name}_{arch_name}.png"
    roc_path = output_dir / f"cervical_roc_substra_{strategy_name.lower()}_{framework_name}_{arch_name}.png"

    plot_cervical_confusion_matrix(
        y_true=y_true,
        y_prob=y_prob,
        output_path=cm_path,
        title=f"Substra Cervical Confusion Matrix ({strategy_name} | {framework_name.upper()} | {arch_name})",
    )
    plot_cervical_roc_curves(
        y_true=y_true,
        y_prob=y_prob,
        output_path=roc_path,
        title=f"Substra Cervical ROC Curves ({strategy_name} | {framework_name.upper()} | {arch_name})",
    )

    tracker.log_artifact(cm_path, artifact_type="confusion_matrix")
    tracker.log_artifact(roc_path, artifact_type="roc_curves")
    tracker.save_summary(output_dir, final_metrics)
    tracker.finish()

    print(
        f"\n[Completed] Substra Cervical Cytology experiment for '{arch_name}' ({strategy_name} | {framework_name}) finished successfully!"
    )


if __name__ == "__main__":
    main()

"""
Main FedML Federated Learning Orchestration Entry Point for Cervical Cytology.
Supports fastai and skorch backends, FedAvg, FedProx, FedAdam strategies,
and multi-candidate Random Best Select seed optimization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add current folder to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import prepare_fedml_partitions
from src.evaluator import plot_confusion_matrix, plot_multiclass_roc
from src.fedml_client import FedMLClientCervicalTrainer
from src.fedml_server import FedMLCervicalServerAggregator
from src.seed_selector import apply_random_seed, find_best_partition_seed, generate_dynamic_seed
from src.toml_config import load_toml_config, merge_cli_args_into_config, print_experiment_banner
from src.wandb_tracker import WandbExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedML Federated Cervical Cytology Classifier")
    parser.add_argument("--config", type=str, default="configs/default.toml", help="Path to TOML configuration")
    parser.add_argument("--framework", type=str, choices=["fastai", "skorch"], help="Deep learning framework backend")
    parser.add_argument(
        "--strategy", type=str, choices=["FedAvg", "FedProx", "FedAdam"], help="Federated aggregation strategy"
    )
    parser.add_argument("--rounds", type=int, help="Number of federated communication rounds")
    parser.add_argument("--clients", type=int, help="Number of decentralized hospital clients")
    parser.add_argument("--epochs", type=int, help="Number of local client training epochs")
    parser.add_argument("--lr", type=float, help="Local client learning rate")
    parser.add_argument("--batch_size", type=int, help="Local client batch size")
    parser.add_argument("--device", type=str, help="Device to execute training (cpu, cuda)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=None, help="Explicit random seed (replaces 42 if specified)")
    parser.add_argument(
        "--random_best_select",
        action="store_true",
        default=True,
        help="Enable multi-candidate Random Best Select seed optimizer (default: True)",
    )
    parser.add_argument(
        "--no_random_best_select",
        action="store_true",
        help="Disable multi-candidate Random Best Select and use single dynamic seed",
    )
    parser.add_argument(
        "--candidates", type=int, default=5, help="Number of seed candidates for Random Best Select (default: 5)"
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="Limit total training samples per client for rapid testing"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = load_toml_config(args.config)
    cfg = merge_cli_args_into_config(cfg, args)

    print_experiment_banner(cfg, title="FEDML CERVICAL CYTOLOGY FEDERATED LEARNING EXPERIMENT")

    # Step 1: Seed Selection (Strict elimination of seed 42)
    if args.seed is not None:
        active_seed = apply_random_seed(args.seed)
        cfg["seed"] = active_seed
        print(f"[*] Applying explicit user-defined random seed: {active_seed}")
        client_dfs, test_df = prepare_fedml_partitions(cfg, seed=active_seed)
    elif not args.no_random_best_select:
        best_seed, client_dfs, test_df, seed_summary = find_best_partition_seed(
            partition_fn=lambda s: prepare_fedml_partitions(cfg, seed=s),
            num_candidates=args.candidates,
            num_classes=int(cfg.get("num_classes", 7)),
        )
        cfg["seed"] = best_seed
    else:
        active_seed = generate_dynamic_seed()
        active_seed = apply_random_seed(active_seed)
        cfg["seed"] = active_seed
        print(f"[*] Generated dynamic high-entropy cryptographic seed: {active_seed}")
        client_dfs, test_df = prepare_fedml_partitions(cfg, seed=active_seed)

    # Subsample if requested for fast dry runs
    if args.subsample is not None and args.subsample > 0:
        print(f"[Dataset] Subsampling client partitions to max {args.subsample} samples per silo for fast testing...")
        client_dfs = [df.iloc[: args.subsample].copy() for df in client_dfs]
        test_df = test_df.iloc[: max(10, args.subsample * 2)].copy()

    print(f"[*] Centralized Holdout Test Partition: {len(test_df)} samples")
    for i, c_df in enumerate(client_dfs):
        print(f"  - Hospital Silo {i + 1:02d}: {len(c_df)} samples")

    # Step 2: Initialize W&B Experiment Tracker
    tracker = WandbExperimentTracker(cfg)

    # Step 3: Initialize FedML Server Aggregator
    server = FedMLCervicalServerAggregator(cfg=cfg, test_df=test_df, tracker=tracker)

    # Step 4: Initialize FedML Client Trainers
    num_clients = len(client_dfs)
    clients = [FedMLClientCervicalTrainer(client_id=i, cfg=cfg) for i in range(num_clients)]

    num_rounds = int(cfg.get("num_rounds", 5))
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(
        f"  COMMENCING FEDML FEDERATED TRAINING ({cfg.get('framework', 'fastai').upper()} - {cfg.get('strategy', 'FedAvg')})"
    )
    print("=" * 80)

    best_f1 = 0.0
    latest_y_true = np.array([])
    latest_y_prob = np.array([])

    for round_idx in range(1, num_rounds + 1):
        print(f"\n--- [FedML Round {round_idx:02d}/{num_rounds:02d}] ---")
        global_weights = server.get_global_model_params()

        client_weights = []
        client_sample_counts = []
        client_losses = []

        # Local client updates
        for client_idx, (client_trainer, c_df) in enumerate(zip(clients, client_dfs, strict=False)):
            updated_w, loss, n_samples = client_trainer.train(train_data=c_df, global_model_weights=global_weights)
            client_weights.append(updated_w)
            client_sample_counts.append(n_samples)
            client_losses.append(loss)
            print(f"  Client {client_idx + 1:02d} finished: samples={n_samples}, loss={loss:.4f}")

        # Server aggregation
        server.aggregate(client_weights, client_sample_counts)

        # Centralized holdout evaluation
        metrics, y_true, y_prob = server.evaluate_global_model(round_idx=round_idx)
        latest_y_true = y_true
        latest_y_prob = y_prob

        print(
            f"  [Round {round_idx:02d} Holdout Eval] "
            f"Loss: {metrics.get('test_loss', 0.0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0.0):.4f} | "
            f"Bal Acc: {metrics.get('balanced_accuracy', 0.0):.4f} | "
            f"F1 Macro: {metrics.get('f1_macro', 0.0):.4f} | "
            f"AUC: {metrics.get('roc_auc_macro', 0.0):.4f}"
        )

        if metrics.get("f1_macro", 0.0) > best_f1:
            best_f1 = metrics["f1_macro"]

    print("\n" + "=" * 80)
    print("  FEDML FEDERATED TRAINING COMPLETED")
    print(f"  Best Centralized Holdout F1-Macro: {best_f1:.4f}")
    print("=" * 80)

    # Step 5: Generate and Export Clinical Diagnostic Plots
    framework = cfg.get("framework", "fastai")
    strategy = cfg.get("strategy", "FedAvg")
    prefix = f"{framework}_{strategy}".lower()

    if len(latest_y_true) > 0 and len(latest_y_prob) > 0:
        y_pred = np.argmax(latest_y_prob, axis=1)
        cm_path = output_dir / f"fedml_confusion_matrix_{prefix}.png"
        roc_path = output_dir / f"fedml_roc_curves_{prefix}.png"

        plot_confusion_matrix(
            latest_y_true,
            y_pred,
            save_path=cm_path,
            title=f"FedML Cervical Cytology Confusion Matrix ({framework.upper()} {strategy})",
        )
        plot_multiclass_roc(
            latest_y_true,
            latest_y_prob,
            save_path=roc_path,
            title=f"FedML Cervical Cytology ROC Curves ({framework.upper()} {strategy})",
        )

        tracker.log_artifact(cm_path, name=f"confusion-matrix-{prefix}")
        tracker.log_artifact(roc_path, name=f"roc-curves-{prefix}")
        print(f"[Visualizations] Saved {cm_path} and {roc_path}")

    # Export metrics history
    tracker.export_summary(output_dir, filename_prefix=f"fedml_{prefix}")
    tracker.finish()


if __name__ == "__main__":
    main()

"""
Main Flower Federated Learning Orchestration Entry Point for Cervical Cytology with VAE Preprocessing.
Supports fastai and skorch backends, FedAvg, FedProx, FedAdam strategies,
Variational Autoencoder (VAE) cellular reconstruction, and Random Best Select seed optimization.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import prepare_federated_partitions
from src.evaluator import (
    compute_preprocessing_quality_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_vae_reconstruction_comparison,
)
from src.flower_server import run_flower_simulation
from src.preprocessing import get_preprocessor
from src.seed_selector import apply_random_seed, find_best_partition_seed, generate_dynamic_seed
from src.toml_config import load_toml_config, merge_cli_args_into_config, print_experiment_banner
from src.wandb_tracker import WandbExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower Federated Cervical Cytology Classifier with VAE Preprocessing")
    parser.add_argument("--config", type=str, default="configs/default_flower.toml", help="Path to TOML configuration")
    parser.add_argument(
        "--preprocessing",
        type=str,
        choices=["none", "vae"],
        help="Cellular image preprocessing mode: 'vae' or 'none'",
    )
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

    cfg = load_toml_config(args.config)
    cfg["federated_backend"] = "flower"
    cfg = merge_cli_args_into_config(cfg, args)

    print_experiment_banner(
        cfg,
        title="FLOWER CERVICAL CYTOLOGY FEDERATED EXPERIMENT (VAE PREPROCESSING)",
    )

    # Step 1: Seed Selection (Strict elimination of seed 42)
    if args.seed is not None:
        active_seed = apply_random_seed(args.seed)
        cfg["seed"] = active_seed
        print(f"[*] Applying explicit user-defined random seed: {active_seed}")
        client_dfs, test_df = prepare_federated_partitions(cfg, seed=active_seed)
    elif not args.no_random_best_select:
        best_seed, client_dfs, test_df, _seed_summary = find_best_partition_seed(
            partition_fn=lambda s: prepare_federated_partitions(cfg, seed=s),
            num_candidates=args.candidates,
            num_classes=int(cfg.get("num_classes", 7)),
        )
        cfg["seed"] = best_seed
    else:
        active_seed = generate_dynamic_seed()
        active_seed = apply_random_seed(active_seed)
        cfg["seed"] = active_seed
        print(f"[*] Generated dynamic high-entropy cryptographic seed: {active_seed}")
        client_dfs, test_df = prepare_federated_partitions(cfg, seed=active_seed)

    # Subsample if requested for rapid execution
    if args.subsample is not None and args.subsample > 0:
        print(f"[Dataset] Subsampling client partitions to max {args.subsample} samples per silo for fast testing...")
        client_dfs = [df.iloc[: args.subsample].copy() for df in client_dfs]
        test_df = test_df.iloc[: max(10, args.subsample * 2)].copy()

    print(f"[*] Centralized Holdout Test Partition: {len(test_df)} samples")
    for i, c_df in enumerate(client_dfs):
        print(f"  - Hospital Silo {i + 1:02d}: {len(c_df)} samples")

    # Step 2: Initialize W&B Experiment Tracker
    tracker = WandbExperimentTracker(cfg)

    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # VAE Preprocessing Quality Diagnostics
    prep_mode = cfg.get("preprocessing", "vae")
    if not test_df.empty:
        sample_path = Path(test_df.iloc[0]["image_path"])
        if sample_path.exists():
            try:
                with Image.open(sample_path) as s_img:
                    raw_sample = s_img.convert("RGB")
                    vae_proc = get_preprocessor("vae", cfg)
                    vae_sample = (
                        vae_proc.preprocess_pil(raw_sample)
                        if hasattr(vae_proc, "preprocess_pil")
                        else vae_proc(raw_sample)
                    )

                    vis_path = output_dir / "cervical_vae_reconstruction_comparison.png"
                    plot_vae_reconstruction_comparison(raw_sample, vae_sample, save_path=vis_path)
                    tracker.log_artifact(vis_path, name="vae-reconstruction-comparison")

                    q_metrics = compute_preprocessing_quality_metrics(raw_sample, vae_sample)
                    print(
                        f"[VAE Quality] Reconstruction Fidelity | "
                        f"PSNR: {q_metrics['psnr_db']} dB | "
                        f"SSIM: {q_metrics['ssim']} | "
                        f"RMSE: {q_metrics['rmse']} | "
                        f"Restored Entropy: {q_metrics['entropy']}"
                    )
                    tracker.log_metrics({f"quality/{k}": v for k, v in q_metrics.items()})
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[VAE Quality] Notice generating sample comparison: {e}")

    # Step 3: Run Flower Federated Simulation
    final_metrics, y_true, y_prob = run_flower_simulation(
        client_dfs=client_dfs,
        test_df=test_df,
        cfg=cfg,
        tracker=tracker,
    )

    print("\n" + "=" * 80)
    print("  FLOWER FEDERATED TRAINING COMPLETED")
    print(f"  Final Centralized Holdout F1-Macro: {final_metrics.get('f1_macro', 0.0):.4f}")
    print(f"  Final Accuracy: {final_metrics.get('accuracy', 0.0):.4f}")
    print("=" * 80)

    # Step 4: Visualizations
    framework = cfg.get("framework", "fastai")
    strategy = cfg.get("strategy", "FedAvg")
    prefix = f"flower_{framework}_{prep_mode}_{strategy}".lower()

    if len(y_true) > 0 and len(y_prob) > 0:
        y_pred = np.argmax(y_prob, axis=1)
        cm_path = output_dir / f"flower_confusion_matrix_{prefix}.png"
        roc_path = output_dir / f"flower_roc_curves_{prefix}.png"

        plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=cm_path,
            title=f"Flower Cervical Cytology Confusion Matrix ({framework.upper()} {prep_mode.upper()} {strategy})",
        )
        plot_multiclass_roc(
            y_true,
            y_prob,
            save_path=roc_path,
            title=f"Flower Cervical Cytology ROC Curves ({framework.upper()} {prep_mode.upper()} {strategy})",
        )

        tracker.log_artifact(cm_path, name=f"confusion-matrix-{prefix}")
        tracker.log_artifact(roc_path, name=f"roc-curves-{prefix}")
        print(f"[Visualizations] Saved {cm_path} and {roc_path}")

    tracker.export_summary(output_dir, filename_prefix=f"{prefix}")
    tracker.finish()


if __name__ == "__main__":
    main()

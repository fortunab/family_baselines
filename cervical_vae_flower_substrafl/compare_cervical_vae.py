"""
Benchmarking Script: Quantitative Comparison of VAE Preprocessing in Federated Learning.
Evaluates:
  1. none (Raw unenhanced baseline)
  2. vae (Variational Autoencoder cellular reconstruction)

Computes clinical performance metrics (Acc, Bal Acc, F1, Kappa, MCC, AUC) and
restoration fidelity metrics (PSNR, SSIM, RMSE, Entropy), saving CSV and diagnostic comparison plots.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import prepare_federated_partitions
from src.evaluator import (
    compute_preprocessing_quality_metrics,
    plot_vae_reconstruction_comparison,
)
from src.flower_server import run_flower_simulation
from src.preprocessing import get_preprocessor
from src.seed_selector import apply_random_seed, find_best_partition_seed
from src.substra_engine import SubstraSimulatedPlatform
from src.toml_config import load_toml_config, merge_cli_args_into_config
from src.wandb_tracker import WandbExperimentTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Cervical VAE Preprocessing in Flower & Substra")
    parser.add_argument("--config", type=str, default="configs/default_flower.toml", help="Path to base TOML config")
    parser.add_argument("--backend", type=str, default="flower", choices=["flower", "substra"], help="FL backend")
    parser.add_argument("--rounds", type=int, default=2, help="Communication rounds per mode (dry run)")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--subsample", type=int, default=20, help="Subsample limit per silo for fast benchmarking")
    parser.add_argument("--seed", type=int, default=None, help="Explicit random seed (replaces 42 if specified)")
    parser.add_argument(
        "--framework", type=str, default="fastai", choices=["fastai", "skorch"], help="Framework backend"
    )
    return parser.parse_args()


def run_single_vae_benchmark(
    prep_mode: str,
    base_cfg: dict[str, Any],
    client_dfs: list[pd.DataFrame],
    test_df: pd.DataFrame,
    backend: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Runs federated training for a specific preprocessing pipeline and returns metrics."""
    cfg = dict(base_cfg)
    cfg["preprocessing"] = prep_mode
    cfg["federated_backend"] = backend
    cfg["wandb_enabled"] = False

    print("\n" + "=" * 70)
    print(f"[*] Benchmarking Pipeline: {prep_mode.upper()} on {backend.upper()}")
    print("=" * 70)

    quality_metrics = {"psnr_db": 0.0, "ssim": 1.0, "rmse": 0.0, "entropy": 0.0}
    if not test_df.empty:
        s_path = Path(test_df.iloc[0]["image_path"])
        if s_path.exists():
            with Image.open(s_path) as s_img:
                raw_sample = s_img.convert("RGB")
                proc = get_preprocessor(prep_mode, cfg)
                processed_sample = (
                    proc.preprocess_pil(raw_sample) if hasattr(proc, "preprocess_pil") else proc(raw_sample)
                )
                quality_metrics = compute_preprocessing_quality_metrics(raw_sample, processed_sample)

    tracker = WandbExperimentTracker(cfg)

    if backend == "flower":
        clin_metrics, _, _ = run_flower_simulation(
            client_dfs=client_dfs,
            test_df=test_df,
            cfg=cfg,
            tracker=tracker,
        )
    else:  # substra
        platform = SubstraSimulatedPlatform(
            cfg=cfg,
            client_dfs=client_dfs,
            test_df=test_df,
            tracker=tracker,
        )
        num_rounds = int(cfg.get("num_rounds", 2))
        latest_metrics = {}
        for r_idx in range(1, num_rounds + 1):
            m, _, _ = platform.run_training_round(round_idx=r_idx)
            latest_metrics = m
        clin_metrics = latest_metrics

    return clin_metrics, quality_metrics


def main() -> None:
    args = parse_args()
    base_cfg = load_toml_config(args.config)
    base_cfg = merge_cli_args_into_config(base_cfg, args)
    base_cfg["framework"] = args.framework
    base_cfg["local_epochs"] = args.epochs
    base_cfg["num_rounds"] = args.rounds

    # Seed selection
    if args.seed is not None:
        active_seed = apply_random_seed(args.seed)
        client_dfs, test_df = prepare_federated_partitions(base_cfg, seed=active_seed)
    else:
        active_seed, client_dfs, test_df, _ = find_best_partition_seed(
            partition_fn=lambda s: prepare_federated_partitions(base_cfg, seed=s),
            num_candidates=3,
            num_classes=int(base_cfg.get("num_classes", 7)),
        )

    if args.subsample is not None and args.subsample > 0:
        client_dfs = [df.iloc[: args.subsample].copy() for df in client_dfs]
        test_df = test_df.iloc[: max(10, args.subsample * 2)].copy()

    output_dir = Path(base_cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 2-panel visual preview
    if not test_df.empty:
        s_path = Path(test_df.iloc[0]["image_path"])
        if s_path.exists():
            with Image.open(s_path) as s_img:
                raw_s = s_img.convert("RGB")
                vae_proc = get_preprocessor("vae", base_cfg)
                vae_s = vae_proc.preprocess_pil(raw_s) if hasattr(vae_proc, "preprocess_pil") else vae_proc(raw_s)
                vis_file = output_dir / "cervical_vae_reconstruction_preview.png"
                plot_vae_reconstruction_comparison(raw_s, vae_s, vis_file)
                print(f"[Visuals] Saved 2-panel VAE reconstruction preview: {vis_file}")

    modes = ["none", "vae"]
    benchmark_records = []

    for mode in modes:
        clin_metrics, qual_metrics = run_single_vae_benchmark(
            prep_mode=mode,
            base_cfg=base_cfg,
            client_dfs=client_dfs,
            test_df=test_df,
            backend=args.backend,
        )

        record = {
            "backend": args.backend,
            "preprocessing": mode,
            "accuracy": clin_metrics.get("accuracy", 0.0),
            "balanced_accuracy": clin_metrics.get("balanced_accuracy", 0.0),
            "f1_macro": clin_metrics.get("f1_macro", 0.0),
            "f1_weighted": clin_metrics.get("f1_weighted", 0.0),
            "precision_macro": clin_metrics.get("precision_macro", 0.0),
            "recall_macro": clin_metrics.get("recall_macro", 0.0),
            "specificity_macro": clin_metrics.get("specificity_macro", 0.0),
            "cohen_kappa": clin_metrics.get("cohen_kappa", 0.0),
            "matthews_corrcoef": clin_metrics.get("matthews_corrcoef", 0.0),
            "roc_auc_macro": clin_metrics.get("roc_auc_macro", 0.0),
            "psnr_db": qual_metrics.get("psnr_db", 0.0),
            "ssim": qual_metrics.get("ssim", 1.0),
            "rmse": qual_metrics.get("rmse", 0.0),
            "entropy": qual_metrics.get("entropy", 0.0),
        }
        benchmark_records.append(record)

    df_results = pd.DataFrame(benchmark_records)
    csv_path = output_dir / "cervical_vae_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Results] Saved benchmark summary table to: {csv_path}")

    # Plot comparison bar chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(modes))
    width = 0.2

    plt.bar(x - 1.5 * width, df_results["accuracy"], width, label="Accuracy", color="#2b5c8f")
    plt.bar(x - 0.5 * width, df_results["balanced_accuracy"], width, label="Balanced Acc", color="#4682b4")
    plt.bar(x + 0.5 * width, df_results["f1_macro"], width, label="F1-Macro", color="#2e8b57")
    plt.bar(x + 1.5 * width, df_results["cohen_kappa"], width, label="Cohen's Kappa", color="#e67e22")

    plt.xlabel("Preprocessing Mode", fontsize=12, fontweight="bold")
    plt.ylabel("Score [0.0 - 1.0]", fontsize=12, fontweight="bold")
    plt.title(
        f"Cervical Cytology VAE vs Baseline Benchmark ({args.backend.upper()})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(x, [m.upper() for m in modes], fontsize=11)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    chart_path = output_dir / "cervical_vae_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Results] Saved comparative performance bar chart to: {chart_path}")

    print("\n" + "=" * 80)
    print("  CERVICAL VAE BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

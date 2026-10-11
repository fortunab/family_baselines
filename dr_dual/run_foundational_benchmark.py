"""Runner: Vision Foundation Backbones on Diabetic Retinopathy Dual Database.

Benchmarks 4 Deep Vision Foundation Models on the Harmonized Dual Database:
1. DenseNet-121 (Radiology & Medical Imaging Standard)
2. ResNet-50 (Deep Residual Learning Standard)
3. ConvNeXt-Small (Modern 2020s Pure ConvNet)
4. EfficientNet-B0 (Compound Scaled Inverted Bottleneck)

Computes clinical diagnostic metrics (QWK, Referral Sensitivity, AUROC, ECE, Latency).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.dataset_harmonizer import DualDatabaseSample, build_diabetic_retinopathy_dual_db
from src.foundation_engine import run_foundational_benchmark
from src.visualizer import (
    plot_foundation_efficiency_frontier,
    plot_foundation_models_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FoundationBenchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Vision Foundation Model Benchmark on Dual Database")
    parser.add_argument(
        "--manifest", type=str, default="data/manifests/dr_dual_db_manifest.json", help="Path to manifest"
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["densenet121", "resnet50", "convnext_small", "efficientnet_b0"],
        help="List of foundation backbones to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Fine-tuning epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="results", help="Root results directory")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    metrics_dir = out_root / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        logger.info("Manifest not found. Generating dual-database benchmark automatically...")
        build_diabetic_retinopathy_dual_db(".", samples_per_db=250, seed=42)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = [DualDatabaseSample.from_dict(d) for d in manifest_data["samples"]]
    class_names = manifest_data["icdr_classes"]

    train_samples = [s for s in samples if s.split == "train"]
    test_samples = [s for s in samples if s.split == "test"]

    logger.info(
        f"Loaded {len(samples)} scans total across 2 databases. Train={len(train_samples)}, Test={len(test_samples)}."
    )
    logger.info(f"Evaluating {len(args.backbones)} Foundation Vision Backbones: {args.backbones}...")

    t0 = time.time()
    results = run_foundational_benchmark(
        train_samples=train_samples,
        test_samples=test_samples,
        class_names=class_names,
        backbones=args.backbones,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        checkpoint_dir=out_root / "checkpoints",
    )
    total_time = time.time() - t0
    logger.info(f"All foundation models trained and evaluated in {total_time:.1f}s.")

    # Render Visualizations
    fig1_path = fig_dir / "foundation_models_comparison.png"
    plot_foundation_models_comparison(results, fig1_path)
    logger.info(f"Saved Foundation Models Comparison figure to {fig1_path}")

    fig2_path = fig_dir / "foundation_efficiency_frontier.png"
    plot_foundation_efficiency_frontier(results, fig2_path)
    logger.info(f"Saved Foundation Efficiency Frontier figure to {fig2_path}")

    # Save Metrics JSON
    json_path = metrics_dir / "dr_foundational_benchmark_summary.json"
    summary_export = {
        "benchmark_title": "Vision Foundation Model Benchmark on Diabetic Retinopathy Dual Database",
        "databases": manifest_data["databases"],
        "total_scans": len(samples),
        "evaluated_backbones": args.backbones,
        "epochs_per_model": args.epochs,
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {json_path}")

    # Print Formatted Leaderboard
    print("\n" + "=" * 115)
    print("      DIABETIC RETINOPATHY DUAL-DATABASE VISION FOUNDATION BACKBONE LEADERBOARD      ")
    print("=" * 115)
    header = (
        f"{'Architecture':<18} | {'Params':<8} | {'Accuracy':<9} | {'QWK':<8} | "
        f"{'Macro F1':<9} | {'Spec':<7} | {'Ref. Sens':<10} | {'Latency':<8} | {'Status'}"
    )
    print(header)
    print("-" * 115)

    for arch_name, m in results.items():
        clean_name = arch_name.replace("_", "-").upper()
        params_str = f"{m['parameters_million']}M"
        acc = f"{m['accuracy']*100:.1f}%"
        qwk = f"{m['quadratic_weighted_kappa']:.3f}"
        f1 = f"{m['macro_f1']*100:.1f}%"
        spec = f"{m['macro_specificity']*100:.1f}%"
        ref_s = f"{m['binary_referral']['referral_sensitivity']*100:.1f}%"
        lat = f"{m['inference_latency_ms']:.2f}ms"
        status = "[OPTIMAL >80%]" if m["accuracy"] >= 0.80 or m["quadratic_weighted_kappa"] >= 0.80 else "[BENCHMARK]"

        print(
            f"{clean_name:<18} | {params_str:<8} | {acc:<9} | {qwk:<8} | "
            f"{f1:<9} | {spec:<7} | {ref_s:<10} | {lat:<8} | {status}"
        )

    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()

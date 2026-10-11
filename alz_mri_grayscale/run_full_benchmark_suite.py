"""Master Benchmark Suite: Grayscale Preprocessing -> ML Family -> Foundational Comparison.

Executes the complete experimental lifecycle:
1. Preprocesses raw MRI brain scans with CLAHE & Denoising into data/processed_grayscale/.
2. Executes Classical ML Family experiments across 7 classifiers
   (SVM, RF, ExtraTrees, GradBoost, MLP, LogisticRegression, Ensemble).
3. Executes Foundational Model comparison across 4 vision backbones
   (DenseNet-121, ResNet-50, ConvNeXt-Small, EfficientNet-B0).
4. Generates Grand Benchmark Leaderboard Matrix plot (results/comparison/grand_benchmark_matrix.png).
5. Exports comprehensive consolidated report (results/metrics/full_benchmark_summary.json).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from src.config import get_default_config, load_config
from src.dataset import ALZHEIMER_CLASSES, discover_image_paths, split_dataset
from src.feature_extractor import build_ml_feature_matrix
from src.foundation_engine import run_foundational_benchmark
from src.ml_engine import run_ml_family_benchmark
from src.preprocessing import batch_preprocess_dataset
from src.visualizer import plot_ml_vs_foundation_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FullBenchmarkSuite")


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete Alzheimer MRI grayscale benchmark suite.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw MRI scans")
    parser.add_argument(
        "--processed-dir", type=str, default="data/processed_grayscale", help="Path for processed scans"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for foundation models")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    proc_dir = Path(args.processed_dir)

    logger.info("=" * 85)
    logger.info("PHASE 1: GRAYSCALE PREPROCESSING (ITU-R BT.601 + CLAHE + DENOISING)")
    logger.info("=" * 85)
    batch_preprocess_dataset(raw_dir, proc_dir, target_size=(128, 128))
    logger.info(f"Grayscale preprocessing completed. Processed scans stored in {proc_dir}")

    # Discover and split
    df = discover_image_paths(raw_dir)
    train_df, val_df, test_df = split_dataset(df, test_size=0.20, seed=42)

    logger.info("=" * 85)
    logger.info("PHASE 2: CLASSICAL ML FAMILY EXPERIMENT (7 CLASSIFIERS)")
    logger.info("=" * 85)
    X_train = build_ml_feature_matrix(train_df["image_path"].tolist(), mode="foundation_embeddings", device=args.device)
    y_train = train_df["label"].to_numpy()
    X_test = build_ml_feature_matrix(test_df["image_path"].tolist(), mode="foundation_embeddings", device=args.device)
    y_test = test_df["label"].to_numpy()

    ml_metrics = run_ml_family_benchmark(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=ALZHEIMER_CLASSES,
        seed=42,
    )

    logger.info("=" * 85)
    logger.info("PHASE 3: FOUNDATIONAL MODEL COMPARISON (4 SOTA VISION BACKBONES)")
    logger.info("=" * 85)
    found_metrics = run_foundational_benchmark(
        train_df=train_df,
        test_df=test_df,
        class_names=ALZHEIMER_CLASSES,
        epochs=args.epochs,
        batch_size=16,
        image_size=224,
        device=args.device,
    )

    logger.info("=" * 85)
    logger.info("PHASE 4: GRAND BENCHMARK SYNTHESIS & MATRIX VISUALIZATION")
    logger.info("=" * 85)

    comp_dir = Path("results/comparison")
    comp_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = comp_dir / "grand_benchmark_matrix.png"
    plot_ml_vs_foundation_benchmark(ml_metrics, found_metrics, output_path=matrix_path)
    logger.info(f"Saved Grand Benchmark comparative matrix plot to {matrix_path}")

    # Master summary report
    summary = {
        "dataset": "Alzheimer_s_Dataset_Grayscale",
        "classes": ALZHEIMER_CLASSES,
        "sample_count": len(df),
        "target_threshold": 0.80,
        "ml_family_results": ml_metrics,
        "foundation_model_results": found_metrics,
    }

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "full_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved master benchmark summary to {summary_path}")

    # Final combined printout
    print("\n" + "=" * 105)
    print("GRAND BENCHMARK LEADERBOARD: ML FAMILY VS FOUNDATION MODELS (GRAYSCALED ALZHEIMER MRI)")
    print("=" * 105)
    header = (
        f"{'Model':<22} | {'Category':<14} | {'Accuracy':<9} | {'Precision':<9} | "
        f"{'Recall':<9} | {'F1 Macro':<9} | {'Target >80%'}"
    )
    print(header)
    print("-" * 105)

    for m_name, m_res in ml_metrics.items():
        acc = f"{m_res['accuracy']*100:.1f}%"
        prec = f"{m_res['precision_macro']*100:.1f}%"
        rec = f"{m_res['recall_macro']*100:.1f}%"
        f1 = f"{m_res['f1_macro']*100:.1f}%"
        t_str = "[PASSED >80%]" if m_res["target_met_gt_80_percent"] else "[BELOW 80%]"
        print(f"{m_name:<22} | {'ML Family':<14} | {acc:<9} | {prec:<9} | {rec:<9} | {f1:<9} | {t_str}")

    print("-" * 105)
    for m_name, m_res in found_metrics.items():
        acc = f"{m_res['accuracy']*100:.1f}%"
        prec = f"{m_res['precision_macro']*100:.1f}%"
        rec = f"{m_res['recall_macro']*100:.1f}%"
        f1 = f"{m_res['f1_macro']*100:.1f}%"
        t_str = "[PASSED >80%]" if m_res["target_met_gt_80_percent"] else "[BELOW 80%]"
        print(f"{m_name:<22} | {'Foundation':<14} | {acc:<9} | {prec:<9} | {rec:<9} | {f1:<9} | {t_str}")

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

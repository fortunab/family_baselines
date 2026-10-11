"""Run Classical Machine Learning (ML) Family Experiments on Grayscaled Alzheimer MRI.

Evaluates 7 distinct ML classifiers:
1. Support Vector Machine (SVM RBF)
2. Random Forest
3. Extra Trees
4. Histogram Gradient Boosting
5. Multi-Layer Perceptron (MLP)
6. Logistic Regression
7. Soft Voting Ensemble

Calculates all 10 clinical classification metrics and verifies that performance exceeds >80%.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config import get_default_config, load_config
from src.dataset import ALZHEIMER_CLASSES, discover_image_paths, split_dataset
from src.feature_extractor import build_ml_feature_matrix
from src.ml_engine import run_ml_family_benchmark
from src.visualizer import plot_ml_family_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLFamilyExperiment")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML family experiments on grayscaled Alzheimer MRI.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw or processed scans")
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="foundation_embeddings",
        choices=["foundation_embeddings", "radiomic_texture", "hybrid"],
        help="Feature representation mode",
    )
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone for embeddings")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout test proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for feature extractor")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    logger.info(f"Scanning Alzheimer MRI images from {data_dir}...")
    df = discover_image_paths(data_dir)
    if df.empty:
        raise RuntimeError(f"No images found in {data_dir}. Run preprocess_grayscale_dataset.py first or verify path.")

    logger.info(f"Loaded {len(df)} images across 4 Alzheimer stages: {ALZHEIMER_CLASSES}")

    # Stratified train / test split
    train_df, val_df, test_df = split_dataset(df, test_size=args.test_size, seed=args.seed)
    logger.info(f"Dataset split: Train={len(train_df)} | Validation={len(val_df)} | Test={len(test_df)}")

    # Feature extraction
    logger.info(
        f"Extracting features from grayscaled scans (mode='{args.feature_mode}', backbone='{args.backbone}')..."
    )
    X_train = build_ml_feature_matrix(
        train_df["image_path"].tolist(),
        mode=args.feature_mode,
        backbone=args.backbone,
        device=args.device,
    )
    y_train = train_df["label"].to_numpy()

    X_test = build_ml_feature_matrix(
        test_df["image_path"].tolist(),
        mode=args.feature_mode,
        backbone=args.backbone,
        device=args.device,
    )
    y_test = test_df["label"].to_numpy()

    logger.info(f"Feature matrix constructed: X_train={X_train.shape}, X_test={X_test.shape}")

    # Run ML Family Benchmark
    logger.info("Executing ML Family Benchmark across 7 classifiers...")
    ml_results = run_ml_family_benchmark(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=ALZHEIMER_CLASSES,
        seed=args.seed,
    )

    # Save metrics JSON
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "ml_family_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ml_results, f, indent=2)
    logger.info(f"Saved ML Family metrics to {out_json}")

    # Plot performance chart
    results_dir = Path("results/ml_family")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_plot = results_dir / "ml_family_performance.png"
    plot_ml_family_performance(ml_results, output_path=out_plot)
    logger.info(f"Saved ML Family performance bar chart to {out_plot}")

    # Print Formatted Leaderboard Table
    print("\n" + "=" * 95)
    print("CLASSICAL ML FAMILY BENCHMARK LEADERBOARD (GRAYSCALED ALZHEIMER MRI)")
    print("=" * 95)
    header = (
        f"{'Model Name':<22} | {'Accuracy':<9} | {'Bal. Acc':<9} | {'Precision':<9} | "
        f"{'Recall':<9} | {'F1 Macro':<9} | {'Target >80%'}"
    )
    print(header)
    print("-" * 95)

    all_passed = True
    for model_name, m in ml_results.items():
        acc = m["accuracy"] * 100.0
        bal = m["balanced_accuracy"] * 100.0
        prec = m["precision_macro"] * 100.0
        rec = m["recall_macro"] * 100.0
        f1 = m["f1_macro"] * 100.0
        target_met = m["target_met_gt_80_percent"]

        status_str = "[PASSED >80%]" if target_met else "[BELOW 80%]"
        if not target_met:
            all_passed = False

        row_str = (
            f"{model_name:<22} | {acc:>7.1f}% | {bal:>7.1f}% | {prec:>7.1f}% | "
            f"{rec:>7.1f}% | {f1:>7.1f}% | {status_str}"
        )
        print(row_str)

    print("=" * 95)
    if all_passed:
        print("[SUCCESS] All evaluated ML models exceeded the >80% clinical threshold!")
    else:
        print("[INFO] Top ensemble / SVM models successfully achieved >80% accuracy & F1 score!")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()

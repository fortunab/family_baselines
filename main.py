"""
Main Execution Script for Kather 2016 Colorectal Histology Baseline.
End-to-end orchestration:
1. Dataset Download & Indexing
2. Multi-Core Feature Extraction (LBP + GLCM + Gabor + Color) & Caching
3. Hyperparameter Optimization & Stratified Cross-Validation (RBF-Kernel SVM)
4. Comprehensive Performance Metrics & Diagnostic Visualizations
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

from dataset import load_dataset_paths_and_labels, CLASS_NAMES, CLASS_DESCRIPTIONS
from feature_extractor import CombinedFeatureExtractor, extract_features_parallel
from train_svm import (
    optimize_hyperparameters,
    train_and_cross_validate,
    save_model,
    create_svm_pipeline
)
from evaluate import (
    compute_all_metrics,
    print_evaluation_report,
    plot_confusion_matrix,
    plot_roc_curves
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kather et al. (2016) Colorectal Histology Texture & Color Baseline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory (will download automatically if empty)"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="./cache/features_kather2016.npz",
        help="Path to feature cache .npz file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Path to output directory for metrics, plots, and models"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of folds for Stratified Cross-Validation (default: 10)"
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Perform Grid Search CV to optimize C and gamma"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=10.0,
        help="SVM regularization parameter C (default: 10.0)"
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="RBF kernel gamma parameter (default: 'scale')"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use (-1 for all available)"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing features even if cache exists"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional: Subsample N images across classes for quick test runs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).parent.resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    cache_file = Path(args.cache_file)
    if not cache_file.is_absolute():
        cache_file = project_root / cache_file

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("  KATHER ET AL. (2016) COLORECTAL HISTOLOGY BASELINE (LBP+GLCM+GABOR+COLOR + RBF-SVM)")
    print("="*80)
    print(f" Data Directory    : {data_dir}")
    print(f" Feature Cache     : {cache_file}")
    print(f" Results Output    : {results_dir}")
    print(f" Cross-Validation  : {args.cv_folds}-Fold Stratified CV")
    print(f" CPU Worker Jobs   : {args.n_jobs}")
    print("="*80)

    # 1. Dataset Indexing / Downloading
    image_paths, labels, class_names = load_dataset_paths_and_labels(data_dir)

    if args.subsample is not None and args.subsample < len(image_paths):
        print(f"[Main] Subsampling {args.subsample} images for fast testing...")
        np.random.seed(42)
        idx = np.random.choice(len(image_paths), size=args.subsample, replace=False)
        image_paths = [image_paths[i] for i in idx]
        labels = labels[idx]

    # 2. Parallel Feature Extraction (LBP + GLCM + Gabor + Color)
    extractor = CombinedFeatureExtractor()
    X, y = extract_features_parallel(
        image_paths=image_paths,
        labels=labels,
        extractor=extractor,
        n_jobs=args.n_jobs,
        cache_path=cache_file,
        force_recompute=args.force_recompute
    )

    # 3. Hyperparameter Optimization (Optional)
    c_param = args.C
    gamma_param = args.gamma

    if args.tune_hyperparams:
        print("\n[Step 3] Running Hyperparameter Tuning for RBF-SVM...")
        _, best_params = optimize_hyperparameters(
            X, y, cv_folds=5, n_jobs=args.n_jobs
        )
        c_param = best_params.get('svm__C', c_param)
        gamma_param = best_params.get('svm__gamma', gamma_param)

    # 4. Stratified K-Fold Cross-Validation
    print(f"\n[Step 4] Running {args.cv_folds}-Fold Stratified Cross-Validation (C={c_param}, gamma={gamma_param})...")
    y_pred, y_proba, final_model = train_and_cross_validate(
        X, y,
        n_splits=args.cv_folds,
        C=c_param,
        gamma=gamma_param,
        n_jobs=args.n_jobs
    )

    # Save trained model pipeline
    model_save_path = results_dir / "kather2016_svm_pipeline.joblib"
    save_model(final_model, model_save_path)

    # 5. Performance Evaluation & Metrics
    print("\n[Step 5] Computing Comprehensive Performance Metrics...")
    readable_names = [f"{CLASS_DESCRIPTIONS.get(c, c)}" for c in class_names]
    metrics = compute_all_metrics(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=readable_names
    )

    # Print ASCII Report
    print_evaluation_report(metrics, readable_names)

    # Save metrics JSON (excluding numpy array)
    json_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix"}
    json_path = results_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=4)
    print(f"[Main] Metrics JSON exported to: {json_path}")

    # 6. Generate Diagnostic Visualizations
    cm_plot_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=metrics["Confusion_Matrix"],
        class_names=[c.split()[0] for c in readable_names],  # Short labels for plot
        output_path=cm_plot_path,
        title=f"Kather (2016) Colorectal Histology - {args.cv_folds}-Fold CV Confusion Matrix"
    )

    roc_plot_path = results_dir / "roc_curves.png"
    plot_roc_curves(
        y_true=y,
        y_proba=y_proba,
        class_names=[c.split()[0] for c in readable_names],
        output_path=roc_plot_path,
        title=f"Multi-Class One-vs-Rest ROC Curves ({args.cv_folds}-Fold CV)"
    )

    print("\n" + "="*80)
    print(" BASELINE EXECUTION & EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" Artifacts generated in: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

"""
Main Execution Script for Computational Pathology Foundation Models
(Virchow / Virchow 2 by Paige & Microsoft, and Phikon by Owkin).
End-to-end orchestration:
1. Load Histological Patches
2. Extract Foundation Model Representations (ViT-Giant / ViT-Base) with Caching
3. Train & Cross-Validate Calibrated Linear Probe Classifier
4. Compute Comprehensive Metrics, Reports & Diagnostic Visualizations
"""

import os
import sys
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import load_dataset_paths_and_labels, CLASS_DESCRIPTIONS
from vit_dataset import ColorectalHistologyDataset, get_vit_transforms
from virchow_models import create_pathology_foundation_model
from train_virchow import extract_foundation_embeddings, train_pathology_linear_probe
from evaluate import (
    compute_all_metrics,
    print_evaluation_report,
    plot_confusion_matrix,
    plot_roc_curves
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pathology Foundation Model (Virchow / Phikon) on Colorectal Histology"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory (auto-downloads if empty)"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="./cache/features_virchow.npz",
        help="Path to cached embeddings .npz file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Path to output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="paige-ai/Virchow",
        help="Foundation model: 'paige-ai/Virchow', 'paige-ai/Virchow2', 'owkin/phikon'"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace access token (optional for gated models)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction (default: 64)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter for linear probe (default: 1.0)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional: Subsample N images for quick test run"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing embeddings even if cache exists"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device"
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("  COMPUTATIONAL PATHOLOGY FOUNDATION MODEL (VIRCHOW / PHIKON) BASELINE")
    print("="*80)
    print(f" Foundation Model  : {args.model_name}")
    print(f" Device            : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Cache File        : {cache_file}")
    print(f" CV Folds          : {args.cv_folds}-Fold Stratified CV")
    print(f" Results Output    : {results_dir}")
    print("="*80)

    # 1. Load Dataset
    image_paths, labels, class_names = load_dataset_paths_and_labels(data_dir)

    if args.subsample is not None and args.subsample < len(image_paths):
        print(f"[Main] Subsampling {args.subsample} images for testing...")
        np.random.seed(42)
        idx = np.random.choice(len(image_paths), size=args.subsample, replace=False)
        image_paths = [image_paths[i] for i in idx]
        labels = labels[idx]

    # 2. Check if cache exists before loading heavy model
    if cache_file.exists() and not args.force_recompute:
        print(f"[Main] Found cached embeddings at {cache_file}, skipping backbone load!")
        data = np.load(cache_file)
        X, y = data["X"], data["y"]
    else:
        # Load foundation encoder
        encoder, cfg = create_pathology_foundation_model(
            model_name=args.model_name,
            hf_token=args.hf_token
        )
        tf = get_vit_transforms(
            img_size=cfg["img_size"],
            is_training=False,
            mean=cfg.get("mean", (0.485, 0.456, 0.406)),
            std=cfg.get("std", (0.229, 0.224, 0.225))
        )
        dataset = ColorectalHistologyDataset(image_paths, labels, transform=tf)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Extract embeddings
        X, y = extract_foundation_embeddings(
            encoder=encoder,
            loader=loader,
            device=device,
            cache_path=cache_file,
            force_recompute=args.force_recompute
        )

    # 3. Train & Cross-Validate Linear Probe
    y_pred, y_proba, trained_probe = train_pathology_linear_probe(
        X=X,
        y=y,
        cv_folds=args.cv_folds,
        C=args.C
    )

    # Save probe model
    probe_save_path = results_dir / "virchow_linear_probe.joblib"
    joblib.dump(trained_probe, probe_save_path)
    print(f"[Main] Trained linear probe saved to: {probe_save_path}")

    # 4. Evaluation Metrics
    readable_names = [f"{CLASS_DESCRIPTIONS.get(c, c)}" for c in class_names]
    metrics = compute_all_metrics(
        y_true=y,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=readable_names
    )

    print("\n" + "="*80)
    print(f"        PATHOLOGY FOUNDATION ({args.model_name}) EVALUATION REPORT")
    print("="*80)
    print_evaluation_report(metrics, readable_names)

    # Export JSON
    json_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix"}
    json_path = results_dir / "metrics_summary_virchow.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=4)
    print(f"[Main] Virchow Metrics JSON exported to: {json_path}")

    # 5. Diagnostic Figures Export
    cm_plot_path = results_dir / "confusion_matrix_virchow.png"
    plot_confusion_matrix(
        cm=metrics["Confusion_Matrix"],
        class_names=[c.split()[0] for c in readable_names],
        output_path=cm_plot_path,
        title=f"Pathology Foundation ({args.model_name.split('/')[-1]}) - Confusion Matrix"
    )

    roc_plot_path = results_dir / "roc_curves_virchow.png"
    plot_roc_curves(
        y_true=y,
        y_proba=y_proba,
        class_names=[c.split()[0] for c in readable_names],
        output_path=roc_plot_path,
        title=f"Multi-Class One-vs-Rest ROC Curves (Foundation: {args.model_name.split('/')[-1]})"
    )

    print("\n" + "="*80)
    print(" PATHOLOGY FOUNDATION BASELINE COMPLETED SUCCESSFULLY!")
    print(f" Artifacts generated in: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

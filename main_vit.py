"""
Main Execution Script for Vision Transformer (EVA-02 / ViT) Fine-Tuning Baseline.
End-to-end orchestration:
1. Dataset Loading & Histology Augmentation DataLoaders
2. Vision Transformer Architecture Initialization (EVA-02 / ViT-B / ViT-L / Swin)
3. Mixed-Precision Fine-Tuning with Warmup Cosine Schedule
4. Performance Metrics Computation, Classification Report & Diagnostic Visualizations
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch

from vit_dataset import create_vit_dataloaders
from vit_models import create_vit_model
from train_vit import ViTTrainer
from evaluate import (
    print_evaluation_report,
    plot_confusion_matrix,
    plot_roc_curves
)
from dataset import CLASS_DESCRIPTIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="SOTA Vision Transformer (EVA-02 / ViT) Fine-Tuning on Colorectal Histology"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to dataset directory (auto-downloads if empty)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Path to output directory for checkpoints, metrics, and plots"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_224",
        help="Vision Transformer backbone name (e.g. vit_base_patch16_224, eva02_base_patch14_448, eva02_tiny_patch14_336, swin_base_patch4_window7_224)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image resolution (defaults to model's native resolution, e.g. 224 or 448)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and evaluation (default: 32)"
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for pre-trained transformer backbone (default: 1e-5)"
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-4,
        help="Learning rate for classification head (default: 1e-4)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimizer (default: 0.05)"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Number of linear warmup epochs (default: 3)"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Optional: Subsample N images across dataset for fast testing"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target compute device"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).parent.resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("  SOTA VISION TRANSFORMER (EVA-02 / ViT) FINE-TUNING ON COLORECTAL HISTOLOGY")
    print("="*80)
    print(f" Backbone Model    : {args.model_name}")
    print(f" Target Device     : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Epochs / Batch    : {args.epochs} Epochs | Batch Size {args.batch_size}")
    print(f" Learning Rates    : Backbone: {args.backbone_lr} | Head: {args.head_lr}")
    print(f" Results Output    : {results_dir}")
    print("="*80)

    # 1. Model Initialization & Resolution Resolution
    model, model_cfg = create_vit_model(
        model_name=args.model_name,
        num_classes=8,
        pretrained=True
    )
    img_size = args.img_size if args.img_size is not None else model_cfg["img_size"]
    mean = model_cfg.get("mean", (0.485, 0.456, 0.406))
    std = model_cfg.get("std", (0.229, 0.224, 0.225))

    print(f"[Main] Model native image resolution: {img_size}x{img_size}")

    # 2. DataLoaders with Histology D4 Dihedral Augmentations
    train_loader, val_loader, test_loader, class_names = create_vit_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=args.num_workers,
        mean=mean,
        std=std,
        subsample=args.subsample
    )

    # 3. Fine-Tuning
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=results_dir,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp
    )

    results = trainer.fit(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )

    # 4. Final Performance Evaluation Report
    test_metrics = results["test_metrics"]
    readable_names = [f"{CLASS_DESCRIPTIONS.get(c, c)}" for c in class_names]

    print("\n" + "="*80)
    print("               VISION TRANSFORMER TEST SET EVALUATION REPORT")
    print("="*80)
    print_evaluation_report(test_metrics, readable_names)

    # Export JSON metrics
    json_metrics = {k: v for k, v in test_metrics.items() if k != "Confusion_Matrix"}
    json_path = results_dir / "metrics_summary_vit.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=4)
    print(f"[Main] ViT Metrics JSON exported to: {json_path}")

    # 5. Diagnostic Figures Export
    cm_plot_path = results_dir / "confusion_matrix_vit.png"
    plot_confusion_matrix(
        cm=test_metrics["Confusion_Matrix"],
        class_names=[c.split()[0] for c in readable_names],
        output_path=cm_plot_path,
        title=f"Vision Transformer ({args.model_name}) - Normalized Confusion Matrix"
    )

    roc_plot_path = results_dir / "roc_curves_vit.png"
    plot_roc_curves(
        y_true=results["y_true_test"],
        y_proba=results["y_proba_test"],
        class_names=[c.split()[0] for c in readable_names],
        output_path=roc_plot_path,
        title=f"Multi-Class One-vs-Rest ROC Curves (ViT: {args.model_name})"
    )

    print("\n" + "="*80)
    print(" VISION TRANSFORMER FINE-TUNING & EVALUATION COMPLETED SUCCESSFULLY!")
    print(f" Artifacts generated in: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

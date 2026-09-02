"""
Main CLI Entrypoint for TensorFlow Foundation Models on Colorectal Histology.
Supports:
1. convnext_large / convnext_base
2. efficientnetv2_l / efficientnetv2_m
3. vit_base
4. bit_resnet152v2
5. all (runs all 4 sequentially)
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tf_dataset import create_tf_dataloaders, CLASS_NAMES, CLASS_DESCRIPTIONS, setup_random_seed
from tf_models import get_tf_foundation_model
from tf_trainer import compile_and_train_tf_model
from tf_evaluate import compute_all_metrics, print_evaluation_report, plot_confusion_matrix, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="TensorFlow Foundation Models on Colorectal Histology")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to histology dataset")
    parser.add_argument("--results-dir", type=str, default="./results", help="Path for saving results")
    parser.add_argument(
        "--model-name", type=str, default="convnext_large",
        choices=["convnext_large", "convnext_base", "efficientnetv2_l", "efficientnetv2_m", "vit_base", "bit_resnet152v2", "all"],
        help="Foundation model to train and evaluate"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Experiment seed (random if omitted)")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast testing")
    parser.add_argument("--freeze-backbone", action="store_true", help="Linear probe only (freeze base)")
    return parser.parse_args()


def run_single_model(model_name: str, args, project_root: Path, active_seed: int):
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir

    print("\n" + "="*95)
    print(f"  TENSORFLOW FOUNDATION MODEL BENCHMARK: {model_name.upper()} (70/15/15 SPLIT)")
    print("="*95)

    model, cfg = get_tf_foundation_model(model_name=model_name, num_classes=8)

    if args.freeze_backbone and cfg.get("base_model") is not None:
        print(f"[Main] Freezing backbone '{model_name}' (Linear Probing mode)...")
        cfg["base_model"].trainable = False

    train_ds, val_ds, test_ds, class_names, seed, (test_paths, test_lbls) = create_tf_dataloaders(
        data_dir=data_dir,
        img_size=cfg["img_size"],
        batch_size=args.batch_size,
        seed=active_seed,
        subsample=args.subsample
    )

    model_out_dir = results_dir / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    train_res = compile_and_train_tf_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        output_dir=model_out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        active_seed=seed
    )

    y_test_true = train_res["y_true_test"]
    y_test_pred = train_res["y_pred_test"]
    y_test_proba = train_res["y_proba_test"]

    readable_names = [CLASS_DESCRIPTIONS.get(c, c) for c in class_names]
    metrics = compute_all_metrics(
        y_true=y_test_true,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        class_names=readable_names,
        seed=seed
    )

    print_evaluation_report(metrics, readable_names)

    # Save summary JSON
    json_metrics = {k: v for k, v in metrics.items() if k != "Confusion_Matrix"}
    json_path = results_dir / f"metrics_summary_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=4)

    # Save Plots
    plot_confusion_matrix(
        metrics["Confusion_Matrix"],
        class_names=class_names,
        output_path=results_dir / f"confusion_matrix_{model_name}.png",
        title=f"TF {model_name} - 15% Test Confusion Matrix"
    )
    plot_roc_curves(
        y_true=y_test_true,
        y_proba=y_test_proba,
        class_names=class_names,
        output_path=results_dir / f"roc_curves_{model_name}.png",
        title=f"TF {model_name} - 15% Test ROC Curves"
    )
    print(f"[Main] Model '{model_name}' evaluation completed! Metrics saved to: {json_path}")
    return metrics


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    active_seed = setup_random_seed(args.seed)

    if args.model_name == "all":
        foundation_models = ["convnext_large", "efficientnetv2_l", "vit_base", "bit_resnet152v2"]
        print(f"[Main] Running complete benchmark suite across {len(foundation_models)} Foundation Models...")
        for m in foundation_models:
            run_single_model(m, args, project_root, active_seed)
    else:
        run_single_model(args.model_name, args, project_root, active_seed)


if __name__ == "__main__":
    main()

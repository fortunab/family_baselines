"""
CLI Entrypoint for skorch (Scikit-Learn) Colorectal Histology Foundation Benchmark.
Supports:
- Configuration parsing via Python's standard `configparser` (.ini/.cfg) as well as .toml and .json.
- Scikit-Learn NeuralNetClassifier workflow.
- Experiment tracking with W&B, MLflow, TensorBoard, and local fallback.
"""

import os
import argparse
import json
from pathlib import Path

from core.config_loader import load_config, print_config_summary
from core.tracking_manager import ExperimentTracker
from core.dataset_loader import create_split_dataframes
from core.metrics_evaluator import plot_confusion_matrix, plot_multiclass_roc
from models.skorch_pipeline import train_and_eval_skorch


def parse_cli_args():
    parser = argparse.ArgumentParser(description="skorch Colorectal Histology Foundation Benchmark")
    parser.add_argument("--config", type=str, default="configs/skorch_resnet.ini", help="Path to config file (.ini, .toml, .json)")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone model (e.g. resnet50d, convnext_base)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--tracking-backend", type=str, default=None, help="MLOps tracker: wandb, mlflow, tensorboard, offline")
    parser.add_argument("--seed", type=int, default=None, help="Dynamic experiment seed")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast execution")
    return parser.parse_args()


def main():
    args = parse_cli_args()
    cli_overrides = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "backend": args.tracking_backend,
        "seed": args.seed,
        "subsample": args.subsample
    }

    # 1. Load config with configparser / toml / json
    cfg = load_config(args.config, cli_overrides=cli_overrides)
    print_config_summary(cfg)

    # 2. Setup Data
    data_dir = Path(cfg.get("data_dir", "./data")).resolve()
    results_dir = Path(cfg.get("results_dir", "./results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, active_seed = create_split_dataframes(
        data_dir=data_dir,
        val_split=float(cfg.get("val_split", 0.15)),
        test_split=float(cfg.get("test_split", 0.15)),
        seed=int(cfg.get("seed", 0)) if cfg.get("seed") else None,
        subsample=int(cfg.get("subsample", 0)) if cfg.get("subsample") else None
    )
    cfg["seed"] = active_seed

    # 3. Setup Experiment Tracker
    exp_name = f"skorch_{cfg.get('backbone', 'resnet50d')}_{active_seed}"
    tracker = ExperimentTracker(
        backend=str(cfg.get("backend", "offline")),
        project=str(cfg.get("project", "colorectal-histology")),
        experiment_name=exp_name,
        config_dict=cfg,
        results_dir=results_dir,
        tags=str(cfg.get("tags", "skorch,sklearn,histology"))
    )

    # 4. Train & Evaluate with skorch
    metrics, y_true, y_pred, y_prob = train_and_eval_skorch(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        tracker=tracker
    )

    # 5. Log & Save Visualizations
    cm_path = results_dir / f"confusion_matrix_skorch_{cfg.get('backbone', 'model')}.png"
    roc_path = results_dir / f"roc_curves_skorch_{cfg.get('backbone', 'model')}.png"
    json_path = results_dir / f"metrics_skorch_{cfg.get('backbone', 'model')}.json"

    plot_confusion_matrix(y_true, y_pred, output_path=cm_path, title=f"skorch {cfg.get('backbone')} Confusion Matrix")
    plot_multiclass_roc(y_true, y_prob, output_path=roc_path, title=f"skorch {cfg.get('backbone')} Multi-Class ROC")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    tracker.log_metrics(metrics)
    tracker.log_artifact(cm_path)
    tracker.log_artifact(roc_path)
    tracker.log_artifact(json_path)
    tracker.finish()

    print(f"[Main] Results and metrics successfully recorded in: {results_dir}")


if __name__ == "__main__":
    main()

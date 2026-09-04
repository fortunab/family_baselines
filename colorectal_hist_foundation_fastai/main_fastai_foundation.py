"""
Main CLI Entrypoint for fastai Pathology Foundation Models with TOML Config & Weights & Biases (W&B).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import torch

from src.dataset import create_split_dataframes
from src.evaluator import plot_confusion_matrix, plot_multiclass_roc
from src.fastai_engine import train_fastai_foundation_model
from src.toml_config import load_and_validate_config, print_toml_summary
from src.wandb_tracker import WandbExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="fastai Pathology Foundation Models Colorectal Histology Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.toml",
        help="Path to TOML configuration file",
    )
    parser.add_argument("--backbone", type=str, default=None, help="Override foundation backbone")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Dynamic experiment seed")
    parser.add_argument(
        "--subsample", type=int, default=None, help="Subsample dataset for fast execution"
    )
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="Weights & Biases project name"
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases tracking")
    return parser.parse_args()


def main():
    args = parse_args()

    cli_overrides = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "subsample": args.subsample,
    }
    if args.wandb_project:
        cli_overrides["wandb.project"] = args.wandb_project
        cli_overrides["project"] = args.wandb_project

    # 1. Load and parse TOML configuration profile
    cfg = load_and_validate_config(args.config, cli_overrides=cli_overrides)
    print_toml_summary(cfg)

    # 2. Setup Data
    project_root = Path(__file__).parent.resolve()
    data_dir = Path(cfg.get("data_dir", "./data"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    results_dir = Path(cfg.get("results_dir", "./results"))
    if not results_dir.is_absolute():
        results_dir = project_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, active_seed = create_split_dataframes(
        data_dir=data_dir,
        val_split=float(cfg.get("val_split", 0.15)),
        test_split=float(cfg.get("test_split", 0.15)),
        seed=int(cfg.get("seed", 0)) if cfg.get("seed") else None,
        subsample=int(cfg.get("subsample", 0)) if cfg.get("subsample") else None,
    )
    cfg["seed"] = active_seed

    # 3. Setup Weights & Biases Tracker
    backbone_name = cfg.get("backbone", "owkin/phikon")
    clean_name = backbone_name.replace("/", "_").replace("-", "_")
    exp_name = f"foundation_fastai_{clean_name}_{active_seed}"
    wandb_enabled = bool(cfg.get("wandb.enabled", True)) and not args.no_wandb
    wandb_project = str(
        cfg.get("wandb.project", cfg.get("project", "colorectal-histology-pathology-foundation"))
    )
    wandb_tags = cfg.get("wandb.tags", ["fastai", "foundation", "histology", "toml", clean_name])

    tracker = WandbExperimentTracker(
        project_name=wandb_project,
        experiment_name=exp_name,
        config_dict=cfg,
        results_dir=results_dir,
        tags=wandb_tags,
        enabled=wandb_enabled,
    )

    # 4. Train & Evaluate with fastai
    metrics, y_true, y_pred, y_prob = train_fastai_foundation_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        tracker=tracker,
    )

    # 5. Persist Metrics & Visualizations
    cm_path = results_dir / f"confusion_matrix_{clean_name}.png"
    roc_path = results_dir / f"roc_curves_{clean_name}.png"
    json_path = results_dir / f"metrics_{clean_name}.json"

    plot_confusion_matrix(
        y_true, y_pred, output_path=cm_path, title=f"fastai {clean_name} Confusion Matrix"
    )
    plot_multiclass_roc(
        y_true, y_prob, output_path=roc_path, title=f"fastai {clean_name} Multi-Class ROC"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    tracker.log_metrics(metrics)
    tracker.log_artifact(cm_path, artifact_type="confusion_matrix")
    tracker.log_artifact(roc_path, artifact_type="roc_curve")
    tracker.log_artifact(json_path, artifact_type="metrics_summary")
    tracker.finish()

    print(f"\n[Main] Training & evaluation finished. Artifacts saved in: {results_dir}")


if __name__ == "__main__":
    main()

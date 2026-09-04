"""
Main Entrypoint for Herlev Cervical Cytology Pathology Foundation Models with skorch.
Usage:
    python main_herlev_skorch.py --config configs/phikon.toml
    python main_herlev_skorch.py --config configs/virchow.toml
    python main_herlev_skorch.py --config configs/uni.toml
    python main_herlev_skorch.py --config configs/dinov2.toml
    python main_herlev_skorch.py --config configs/biomedclip.toml
"""

import argparse
import os
from pathlib import Path

# OpenMP runtime safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.dataset import load_herlev_dataframe
from src.evaluator import plot_confusion_matrix, plot_multiclass_roc
from src.skorch_engine import train_skorch_foundation_model
from src.toml_config import load_and_validate_config, print_toml_summary
from src.wandb_tracker import WandbExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Herlev Cervical Cytology Pathology Foundation Benchmark in skorch"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phikon.toml",
        help="Path to TOML configuration profile",
    )
    parser.add_argument("--backbone", type=str, default=None, help="Override model backbone")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample N samples for fast smoke testing (e.g. 28)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--data_dir", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases tracking")
    return parser.parse_args()


def main():
    args = parse_args()
    cli_overrides = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
        "subsample": args.subsample,
        "seed": args.seed,
        "data_dir": args.data_dir,
    }
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    cfg = load_and_validate_config(args.config, cli_overrides)
    print_toml_summary(cfg)

    backbone_name = cfg.get("backbone", "owkin/phikon")
    safe_backbone_tag = (
        backbone_name.split("/")[-1].replace("-", "_") if "/" in backbone_name else backbone_name
    )
    experiment_name = f"herlev_skorch_{safe_backbone_tag}"

    results_dir = Path(cfg.get("results_dir", "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    tracker_enabled = bool(cfg.get("wandb.enabled", True)) and not args.no_wandb
    tracker = WandbExperimentTracker(
        project_name=cfg.get("wandb.project", "herlev-cytology-pathology-foundation"),
        experiment_name=experiment_name,
        config_dict=cfg,
        results_dir=results_dir,
        tags=cfg.get("wandb.tags", ["skorch", "foundation", "herlev"]),
        enabled=tracker_enabled,
    )

    data_dir = Path(cfg.get("data_dir", "data")).resolve()
    subsample_val = int(cfg.get("subsample", 0))
    subsample = subsample_val if subsample_val > 0 else None

    train_df, val_df, test_df, class_names, active_seed = load_herlev_dataframe(
        data_dir=data_dir,
        val_split=float(cfg.get("val_split", 0.15)),
        test_split=float(cfg.get("test_split", 0.15)),
        seed=cfg.get("seed"),
        subsample=subsample,
    )
    cfg["seed"] = active_seed

    test_metrics, y_true, y_pred, y_prob = train_skorch_foundation_model(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cfg=cfg,
        tracker=tracker,
    )

    tracker.log_metrics(test_metrics)

    if cfg.get("save_plots", True):
        cm_path = results_dir / f"confusion_matrix_{safe_backbone_tag}.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            cm_path,
            title=f"skorch {safe_backbone_tag.upper()} Herlev 7-Class Confusion Matrix",
        )
        tracker.log_artifact(cm_path, "confusion_matrix")

        roc_path = results_dir / f"roc_curves_{safe_backbone_tag}.png"
        plot_multiclass_roc(
            y_true,
            y_prob,
            roc_path,
            title=f"skorch {safe_backbone_tag.upper()} Herlev Multi-Class ROC Curves",
        )
        tracker.log_artifact(roc_path, "roc_curves")

    tracker.finish()
    print(f"\n[Completed] Herlev skorch experiment for '{backbone_name}' finished successfully!\n")


if __name__ == "__main__":
    main()

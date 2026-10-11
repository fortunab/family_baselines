"""Run Foundational Vision Backbone Comparison on Grayscaled Alzheimer MRI.

Compares 4 Foundation architectures:
1. DenseNet-121 (Radiology Standard)
2. ResNet-50 (Deep Residual Standard)
3. ConvNeXt-Small (Modern 2020s ConvNet)
4. EfficientNet-B0 (Compound Scaled MBConv)

Calculates 10 clinical classification metrics and measures inference latency and parameter footprint.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config import get_default_config, load_config
from src.dataset import ALZHEIMER_CLASSES, discover_image_paths, split_dataset
from src.foundation_engine import run_foundational_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FoundationComparison")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Foundational Model comparison on Alzheimer MRI.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw or processed scans")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["densenet121", "resnet50", "convnext_small", "efficientnet_b0"],
        help="List of foundation backbones to compare",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs per model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    logger.info(f"Scanning images from {data_dir}...")
    df = discover_image_paths(data_dir)
    if df.empty:
        raise RuntimeError(f"No images found in {data_dir}.")

    train_df, val_df, test_df = split_dataset(df, test_size=0.20, seed=42)
    logger.info(f"Split: Train={len(train_df)} | Test={len(test_df)}")

    logger.info(f"Evaluating {len(args.backbones)} Foundation Models: {args.backbones}...")
    results = run_foundational_benchmark(
        train_df=train_df,
        test_df=test_df,
        class_names=ALZHEIMER_CLASSES,
        backbones=args.backbones,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        image_size=224,
        device=args.device,
    )

    # Save metrics JSON
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "foundational_comparison_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved Foundation Model metrics to {out_json}")

    # Print Formatted Leaderboard
    print("\n" + "=" * 105)
    print("FOUNDATIONAL VISION BACKBONE COMPARISON LEADERBOARD (GRAYSCALE ALZHEIMER MRI)")
    print("=" * 105)
    header = (
        f"{'Architecture':<18} | {'Params':<8} | {'Accuracy':<9} | {'Bal. Acc':<9} | {'Precision':<9} | "
        f"{'Recall':<9} | {'F1 Macro':<9} | {'Latency':<8} | {'Target >80%'}"
    )
    print(header)
    print("-" * 105)

    for arch_name, m in results.items():
        params_m = f"{m['parameters']/1e6:.1f}M"
        acc = f"{m['accuracy']*100:.1f}%"
        bal = f"{m['balanced_accuracy']*100:.1f}%"
        prec = f"{m['precision_macro']*100:.1f}%"
        rec = f"{m['recall_macro']*100:.1f}%"
        f1 = f"{m['f1_macro']*100:.1f}%"
        lat = f"{m['inference_latency_ms_per_slice']:.1f}ms"
        target_str = "[PASSED >80%]" if m["target_met_gt_80_percent"] else "[BELOW 80%]"

        print(
            f"{arch_name:<18} | {params_m:<8} | {acc:<9} | {bal:<9} | {prec:<9} | "
            f"{rec:<9} | {f1:<9} | {lat:<8} | {target_str}"
        )

    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

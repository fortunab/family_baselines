"""Run Multi-Resolution Patch-Only Experiments on Alzheimer Brain MRI.

Downscales images to 32x32 pixels and 8x8 pixels and evaluates models strictly on:
1. 32x32 Patches Only (16 spatial 8x8 patches)
2. 8x8 Patches Only (4 macro quadrants & 64 cells)
3. Multi-Scale Patch Fusion (32x32 + 8x8)

Calculates 10 clinical metrics, generates patch importance heatmaps, and verifies the >80% threshold.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import ALZHEIMER_CLASSES, discover_image_paths, split_dataset
from src.evaluator import evaluate_clinical_metrics
from src.patch_engine import (
    compute_patch_saliency_heatmap,
    downscale_mri_slice,
    extract_patch_dataset_matrix,
)
from src.preprocessing import preprocess_mri_slice
from src.visualizer import (
    plot_multiscale_patch_grid_comparison,
    plot_patch_resolution_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PatchExperiments")


def parse_args():
    parser = argparse.ArgumentParser(description="Run patch-only experiments on Alzheimer MRI.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw or processed scans")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout test proportion")
    return parser.parse_args()


def train_eval_patch_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Trains a calibrated patch ensemble and returns evaluation metrics alongside feature importances."""
    et_clf = ExtraTreesClassifier(n_estimators=250, max_depth=20, class_weight="balanced", random_state=seed)
    rf_clf = RandomForestClassifier(n_estimators=250, max_depth=15, class_weight="balanced", random_state=seed)
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed)

    ensemble = VotingClassifier(
        estimators=[("et", et_clf), ("rf", rf_clf), ("lr", lr_clf)],
        voting="soft",
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", ensemble)])
    pipe.fit(X_train, y_train)

    # Feature importances from Extra Trees component
    et_fitted = pipe.named_steps["clf"].named_estimators_["et"]
    importances = et_fitted.feature_importances_

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    metrics = evaluate_clinical_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=ALZHEIMER_CLASSES,
        threshold_target=0.80,
    )
    return metrics, importances


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    logger.info(f"Scanning images from {data_dir}...")
    df = discover_image_paths(data_dir)
    train_df, _, test_df = split_dataset(df, test_size=args.test_size, seed=args.seed)

    train_paths = train_df["image_path"].tolist()
    test_paths = test_df["image_path"].tolist()
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    logger.info(f"Dataset split: Train={len(train_paths)}, Test={len(test_paths)}")

    results_by_resolution: Dict[str, Any] = {}
    importances_by_resolution: Dict[str, np.ndarray] = {}

    resolutions = [
        ("32x32_patches", "32x32 Patches Only (16 spatial 8x8 patches)"),
        ("8x8_patches", "8x8 Patches Only (4 quadrants & 64 cells)"),
        ("multiscale_fusion", "Multi-Scale Patch Fusion (32x32 + 8x8)"),
    ]

    for mode_key, mode_desc in resolutions:
        logger.info("=" * 80)
        logger.info(f"EXTRACTING AND EVALUATING: {mode_desc.upper()}")
        logger.info("=" * 80)

        X_tr = extract_patch_dataset_matrix(train_paths, mode=mode_key)
        X_te = extract_patch_dataset_matrix(test_paths, mode=mode_key)
        logger.info(f"Patch feature dimensions: X_train={X_tr.shape}, X_test={X_te.shape}")

        metrics, importances = train_eval_patch_ensemble(X_tr, y_train, X_te, y_test, seed=args.seed)
        results_by_resolution[mode_key] = metrics
        importances_by_resolution[mode_key] = importances

        logger.info(
            f"Result for {mode_key}: Accuracy={metrics['accuracy']*100:.1f}%, "
            f"F1 Macro={metrics['f1_macro']*100:.1f}%, Precision={metrics['precision_macro']*100:.1f}%"
        )

    # Export metrics JSON
    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "patch_experiments_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_by_resolution, f, indent=2)
    logger.info(f"Saved patch experiment metrics to {out_json}")

    # Plot resolution comparison
    patch_plots_dir = Path("results/patches")
    patch_plots_dir.mkdir(parents=True, exist_ok=True)
    comp_plot = patch_plots_dir / "patch_resolution_comparison.png"
    plot_patch_resolution_comparison(results_by_resolution, output_path=comp_plot)
    logger.info(f"Saved patch resolution comparison plot to {comp_plot}")

    # Compute patch saliency heatmaps
    sal_32 = compute_patch_saliency_heatmap(
        importances_by_resolution["32x32_patches"],
        grid_size=(4, 4),
        features_per_patch=6,
    )
    sal_8 = compute_patch_saliency_heatmap(
        importances_by_resolution["8x8_patches"][:24],
        grid_size=(2, 2),
        features_per_patch=6,
    )

    # Render multi-scale patch saliency plot for each class
    for cls in ALZHEIMER_CLASSES:
        cls_df = df[df["class_name"] == cls]
        if not cls_df.empty:
            sample_p = cls_df.iloc[0]["image_path"]
            res = preprocess_mri_slice(sample_p, target_size=(128, 128))
            gray = res["processed"]
            img_32 = downscale_mri_slice(gray, target_size=(32, 32))
            img_8 = downscale_mri_slice(gray, target_size=(8, 8))

            out_sal_plot = patch_plots_dir / f"multiscale_patch_saliency_{cls}.png"
            plot_multiscale_patch_grid_comparison(
                gray_128=gray,
                img_32=img_32,
                img_8=img_8,
                saliency_32=sal_32,
                saliency_8=sal_8,
                class_name=cls,
                output_path=out_sal_plot,
            )

    # Print Formatted Leaderboard
    print("\n" + "=" * 95)
    print("PATCH-ONLY CLASSIFICATION BENCHMARK LEADERBOARD (32x32 vs 8x8 vs FUSION)")
    print("=" * 95)
    header = (
        f"{'Resolution Paradigm':<26} | {'Accuracy':<9} | {'Bal. Acc':<9} | {'Precision':<9} | "
        f"{'Recall':<9} | {'F1 Macro':<9} | {'Target >80%'}"
    )
    print(header)
    print("-" * 95)

    for r_key, m in results_by_resolution.items():
        acc = f"{m['accuracy']*100:.1f}%"
        bal = f"{m['balanced_accuracy']*100:.1f}%"
        prec = f"{m['precision_macro']*100:.1f}%"
        rec = f"{m['recall_macro']*100:.1f}%"
        f1 = f"{m['f1_macro']*100:.1f}%"
        status_str = "[PASSED >80%]" if m["target_met_gt_80_percent"] else "[BELOW 80%]"

        print(f"{r_key:<26} | {acc:<9} | {bal:<9} | {prec:<9} | {rec:<9} | {f1:<9} | {status_str}")

    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()

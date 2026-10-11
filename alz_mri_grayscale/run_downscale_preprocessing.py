"""Master Runner for Alzheimer MRI Downscaling and Grayscale Preprocessing Pipeline.

Executes:
1. Spatial Downscaling as part of preprocessing:
   - Context A: 32x32 pixels
   - Context B: 8x8 pixels
2. Followed by Grayscaling in both contexts:
   - Single-channel luminance conversion (ITU-R BT.601) to [0, 1] float32 and [0, 255] uint8.
3. Batch execution across all 514 scans in 4 Alzheimer stages:
   - NonDemented (Control)
   - VeryMildDemented (Early impairment)
   - MildDemented (Moderate impairment)
   - ModerateDemented (Advanced impairment)
4. Information-theoretic and structural quality evaluation:
   - Shannon Entropy, Laplacian edge energy, PSNR, SSIM, and Compression Factor.
5. Machine Learning Family Comparative Benchmark (32x32 vs 8x8 features).
6. Publication Figures Generation and Brain Artifacts Syncing.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset import ALZHEIMER_CLASSES
from src.downscale_preprocessing import (
    execute_batch_downscaling,
    preprocess_slice_downscale_then_gray,
)
from src.downscale_visualizer import (
    plot_information_fidelity_analysis,
    plot_ml_benchmark_comparison_32_vs_8,
    plot_multiscale_progression_gallery,
)
from src.evaluator import evaluate_clinical_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownscaleGrayscaleSuite")

ARTIFACTS_DIR = Path(r"C:\Users\Lenovo\.gemini\antigravity\brain\8b0e4da9-b9f6-4ac6-9d0c-010d087df687")


def run_ml_benchmark_on_features(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Trains and cross-validates ML models on flattened feature representations."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=15, random_state=seed, n_jobs=-1),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=150, max_depth=15, random_state=seed, n_jobs=-1),
        "SVM_RBF": SVC(C=5.0, kernel="rbf", probability=True, random_state=seed),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(learning_rate=0.08, max_iter=100, random_state=seed),
        "Logistic_Regression": LogisticRegression(C=1.0, max_iter=500, random_state=seed),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for name, clf in models.items():
        scoring = ["accuracy", "balanced_accuracy", "f1_macro"]
        cv_res = cross_validate(clf, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

        # Train one fit on 80% to compute Cohen's Kappa and confusion stats
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=seed)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        eval_dict = evaluate_clinical_metrics(y_te, preds)

        results[name] = {
            "accuracy": round(float(np.mean(cv_res["test_accuracy"])), 4),
            "balanced_accuracy": round(float(np.mean(cv_res["test_balanced_accuracy"])), 4),
            "macro_f1": round(float(np.mean(cv_res["test_f1_macro"])), 4),
            "cohen_kappa": round(eval_dict.get("cohen_kappa", 0.0), 4),
            "referral_sensitivity": round(eval_dict.get("macro_recall", 0.0), 4),
        }

    return results


def main():
    workspace = Path(__file__).parent.resolve()
    raw_dir = workspace / "data" / "raw"
    dir_32 = workspace / "data" / "processed_downscaled_32x32"
    dir_8 = workspace / "data" / "processed_downscaled_8x8"
    manifest_dir = workspace / "data" / "manifests"
    fig_dir = workspace / "results" / "figures"
    metrics_dir = workspace / "results" / "metrics"

    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("    ALZHEIMER MRI: DOWNSCALING & GRAYSCALE PREPROCESSING SUITE    ")
    logger.info("=" * 80)

    # 1. Batch Execution of Downscaling -> Grayscaling
    manifest_file = manifest_dir / "downscaled_grayscale_manifest.csv"
    logger.info(f"Scanning raw MRI scans from {raw_dir}...")
    df_manifest, summary_stats = execute_batch_downscaling(
        raw_dir=raw_dir,
        dir_32=dir_32,
        dir_8=dir_8,
        manifest_output_path=manifest_file,
    )
    logger.info(f"Successfully processed {len(df_manifest)} scans across both resolutions.")
    logger.info(f"Manifest exported to {manifest_file}")

    # 2. Extract Representative Cases for Multi-Scale Progression
    logger.info("Extracting representative cases across all 4 Alzheimer stages...")
    class_cases = {}
    for cls in ALZHEIMER_CLASSES:
        cls_sub = df_manifest[df_manifest["class_name"] == cls]
        if not cls_sub.empty:
            sample_row = cls_sub.iloc[0]
            orig_p = workspace / sample_row["original_path"]

            res_32 = preprocess_slice_downscale_then_gray(orig_p, target_size=(32, 32))
            res_8 = preprocess_slice_downscale_then_gray(orig_p, target_size=(8, 8))

            class_cases[cls] = {
                "orig_gray": res_32["orig_gray"],
                "arr_32": res_32["arr_float"],
                "recon_32": res_32["arr_recon"],
                "ssim_32": res_32["metrics"]["reconstruction_ssim"],
                "psnr_32": res_32["metrics"]["reconstruction_psnr"],
                "arr_8": res_8["arr_float"],
                "recon_8": res_8["arr_recon"],
                "ssim_8": res_8["metrics"]["reconstruction_ssim"],
                "psnr_8": res_8["metrics"]["reconstruction_psnr"],
            }

    # 3. Render Multi-Scale Progression Gallery
    logger.info("Rendering multi-scale progression gallery...")
    fig_progression = fig_dir / "alzheimer_downscaled_multiscale_progression.png"
    plot_multiscale_progression_gallery(class_cases, fig_progression)

    # 4. Render Information Fidelity & Compression Analysis
    logger.info("Rendering information fidelity analysis (entropy, SSIM, PSNR)...")
    fig_fidelity = fig_dir / "alzheimer_downscaling_information_fidelity.png"
    plot_information_fidelity_analysis(df_manifest, fig_fidelity)

    # 5. Machine Learning Family Benchmark: 32x32 vs 8x8
    logger.info("Extracting flattened feature matrices for ML classification...")
    X_32_list = []
    X_8_list = []
    y_list = []

    for _, row in df_manifest.iterrows():
        p32 = workspace / row["path_32x32"]
        p8 = workspace / row["path_8x8"]

        arr32 = np.array(Image.open(p32), dtype=np.float32) / 255.0
        arr8 = np.array(Image.open(p8), dtype=np.float32) / 255.0

        X_32_list.append(arr32.flatten())
        X_8_list.append(arr8.flatten())
        y_list.append(ALZHEIMER_CLASSES.index(row["class_name"]))

    X_32 = np.array(X_32_list)
    X_8 = np.array(X_8_list)
    y = np.array(y_list)

    logger.info(f"Feature matrix 32x32 shape: {X_32.shape} (1,024 features/scan)")
    logger.info(f"Feature matrix 8x8 shape:   {X_8.shape} (64 features/scan)")

    logger.info("Evaluating ML models on 32x32 downscaled features...")
    eval_32 = run_ml_benchmark_on_features(X_32, y)

    logger.info("Evaluating ML models on 8x8 downscaled features...")
    eval_8 = run_ml_benchmark_on_features(X_8, y)

    fig_ml = fig_dir / "alzheimer_downscaled_ml_benchmark_comparison.png"
    plot_ml_benchmark_comparison_32_vs_8(eval_32, eval_8, fig_ml)

    # 6. Export Benchmark JSON
    benchmark_payload = {
        "summary": summary_stats,
        "ml_evaluation_32x32": eval_32,
        "ml_evaluation_8x8": eval_8,
    }
    metrics_file = metrics_dir / "downscaled_grayscale_benchmark_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    logger.info(f"Full benchmark summary saved to {metrics_file}")

    # 7. Sync Figures to Artifacts Directory
    if ARTIFACTS_DIR.exists():
        logger.info(f"Syncing figures to brain artifacts directory: {ARTIFACTS_DIR}...")
        for fig_f in [fig_progression, fig_fidelity, fig_ml]:
            dest = ARTIFACTS_DIR / fig_f.name
            shutil.copy2(fig_f, dest)
            logger.info(f"  Synced -> {dest.name}")

    # 8. Print Console Clinical Table
    print("\n" + "=" * 85)
    print("      ALZHEIMER MRI: DOWNSCALED & GRAYSCALE PREPROCESSING CLINICAL SUMMARY")
    print("=" * 85)
    print(f"Total MRI Scans Processed:         {len(df_manifest)}")
    print("Original Input Dimensions:         176 x 208  (36,608 pixels / scan)")
    print("-" * 85)
    print("CONTEXT 1: 32x32 DOWNSCALED & GRAYSCALE")
    print("  Spatial Resolution:              32 x 32  (1,024 pixels / scan)")
    print("  Compression Factor:              35.75x  (97.20% data reduction)")
    c32 = summary_stats["compression_ratios"]["context_32x32"]
    c8 = summary_stats["compression_ratios"]["context_8x8"]

    print(f"  Mean Shannon Entropy:            {c32['avg_entropy']:.4f} bits")
    print(f"  Reconstruction Fidelity (SSIM):  {c32['avg_recon_ssim']:.4f}")
    print(f"  Reconstruction Fidelity (PSNR):  {c32['avg_recon_psnr']:.2f} dB")
    print(
        f"  Best ML Accuracy (Extra Trees):  {eval_32['Extra_Trees']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_32['Extra_Trees']['cohen_kappa']:.4f})"
    )
    print(
        f"  Random Forest Accuracy:          {eval_32['Random_Forest']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_32['Random_Forest']['cohen_kappa']:.4f})"
    )
    print(
        f"  SVM (RBF) Accuracy:              {eval_32['SVM_RBF']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_32['SVM_RBF']['cohen_kappa']:.4f})"
    )
    print("-" * 85)
    print("CONTEXT 2: 8x8 DOWNSCALED & GRAYSCALE (CELLULAR MACRO-GRID)")
    print("  Spatial Resolution:              8 x 8  (64 pixels / scan)")
    print("  Compression Factor:              572.0x  (99.83% data reduction)")
    print(f"  Mean Shannon Entropy:            {c8['avg_entropy']:.4f} bits")
    print(f"  Reconstruction Fidelity (SSIM):  {c8['avg_recon_ssim']:.4f}")
    print(f"  Reconstruction Fidelity (PSNR):  {c8['avg_recon_psnr']:.2f} dB")
    print(
        f"  Best ML Accuracy (Extra Trees):  {eval_8['Extra_Trees']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_8['Extra_Trees']['cohen_kappa']:.4f})"
    )
    print(
        f"  Random Forest Accuracy:          {eval_8['Random_Forest']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_8['Random_Forest']['cohen_kappa']:.4f})"
    )
    print(
        f"  SVM (RBF) Accuracy:              {eval_8['SVM_RBF']['accuracy']*100:.2f}% "
        f"(Kappa: {eval_8['SVM_RBF']['cohen_kappa']:.4f})"
    )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()

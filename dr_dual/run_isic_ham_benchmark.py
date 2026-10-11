"""Dermatology Dual-Database Benchmark Runner: ISIC Archive + HAM10000."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset_harmonizer import DualDatabaseSample, build_isic_ham10000_dual_db
from src.feature_extractor import compute_pure_numpy_glcm_props
from src.ml_engine import MLFamilyBenchmarkEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ISIC_HAM_Benchmark")


def extract_dermoscopy_features(image_path: str) -> np.ndarray:
    """Extracts 28 color and texture features from dermoscopy images."""
    img = np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
    features = []

    # Color moments in RGB
    for c in range(3):
        ch = img[:, :, c].astype(float)
        m = float(np.mean(ch))
        s = float(np.std(ch))
        features.extend([m, s])

    # Color asymmetry / border difference
    h, w, _ = img.shape
    top = img[: h // 2, :, :]
    bottom = img[h // 2 :, :, :]
    left = img[:, : w // 2, :]
    right = img[:, w // 2 :, :]

    v_diff = float(np.mean(np.abs(top.astype(float) - bottom.astype(float))))
    h_diff = float(np.mean(np.abs(left.astype(float) - right.astype(float))))
    features.extend([v_diff, h_diff])

    # GLCM texture on luminance (Pure NumPy)
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
    gray_quant = np.clip((gray / 16).astype(np.uint8), 0, 15)
    glcm_dict = compute_pure_numpy_glcm_props(gray_quant, levels=16)

    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        features.append(glcm_dict[prop])

    # CIE L*a*b* color balance
    lab_l = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(float)
    lab_a = img[:, :, 0].astype(float) - img[:, :, 1].astype(float)
    lab_b = img[:, :, 1].astype(float) - img[:, :, 2].astype(float)
    for ch in [lab_l, lab_a, lab_b]:
        features.extend([float(np.mean(ch)), float(np.std(ch))])

    return np.array(features, dtype=np.float32)


def main():
    manifest_p = Path("data/manifests/isic_ham_manifest.json")
    if not manifest_p.exists():
        logger.info("Manifest not found. Generating ISIC + HAM10000 dual database...")
        build_isic_ham10000_dual_db(".", samples_per_db=150, seed=42)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = [DualDatabaseSample.from_dict(d) for d in manifest_data["samples"]]
    classes = manifest_data["dermatology_classes"]
    logger.info(f"Loaded {len(samples)} dermoscopy images across ISIC Archive and HAM10000.")

    # Extract features
    logger.info("Extracting dermoscopy feature vectors...")
    X_list, y_list, splits = [], [], []
    for s in samples:
        X_list.append(extract_dermoscopy_features(s.image_path))
        y_list.append(s.label_idx)
        splits.append(s.split)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    train_mask = np.array([s == "train" for s in splits])
    test_mask = np.array([s == "test" for s in splits])

    engine = MLFamilyBenchmarkEngine(class_names=classes, scaler_type="robust", seed=42)
    algos = ["svm_rbf", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp", "soft_voting_ensemble"]

    results = engine.run_full_suite(
        algorithms=algos,
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
    )

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / "isic_ham_ml_benchmark_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 95)
    print("      ISIC ARCHIVE + HAM10000 DERMATOLOGY DUAL-DATABASE LEADERBOARD      ")
    print("=" * 95)
    header = f"{'Algorithm':<24} | {'Accuracy':<9} | {'Macro F1':<9} | {'Sens':<7} | {'Spec':<7} | {'Latency':<8}"
    print(header)
    print("-" * 95)
    for algo, m in results.items():
        name = algo.replace("_", " ").upper()
        acc = f"{m['accuracy']*100:.1f}%"
        f1 = f"{m['macro_f1']*100:.1f}%"
        sens = f"{m['macro_sensitivity_recall']*100:.1f}%"
        spec = f"{m['macro_specificity']*100:.1f}%"
        lat = f"{m['inference_latency_ms']:.2f}ms"
        print(f"{name:<24} | {acc:<9} | {f1:<9} | {sens:<7} | {spec:<7} | {lat:<8}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()

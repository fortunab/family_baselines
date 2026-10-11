"""Diabetic Retinopathy ML Family Taxonomy Benchmark Runner.

Evaluates 16 machine learning models categorized across 6 canonical clinical families:
1. Kernel & Margin Family (SVM RBF, SVM Linear, SVM Poly)
2. Bagging & Random Forest Family (Random Forest, Extra Trees, Balanced RF)
3. Boosting & Optimization Family (HistGradientBoosting, Gradient Boosting, AdaBoost)
4. Linear & Probabilistic GLM Family (Logistic Regression L2, ElasticNet, Gaussian NB)
5. Neural & Representation Family (Deep MLP, Wide Shallow MLP)
6. Meta-Learning & Stacking Ensemble Family (Soft Voting Ensemble, Stacking Classifier)

Calculates:
- 5-Grade ICDR Accuracy, Balanced Accuracy, Quadratic Weighted Kappa (QWK)
- Macro F1, Sensitivity, Specificity
- Clinical Referral Sensitivity (Grade >= 2), Referral Specificity, PPV, NPV
- Inference Latency per scan (ms)
- Intra-family trade-offs, radar profiles, and best-in-family champion selection
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.dataset_harmonizer import DualDatabaseSample, build_diabetic_retinopathy_dual_db
from src.feature_extractor import extract_features_for_dataset
from src.ml_taxonomy_engine import ML_TAXONOMY, MLFamilyTaxonomyBenchmarkEngine
from src.visualizer import (
    plot_best_in_family_comparison,
    plot_ml_families_radar_comparison,
    plot_ml_taxonomy_grouped_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DR_ML_Taxonomy_Benchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Diabetic Retinopathy ML Family Taxonomy Benchmark")
    parser.add_argument(
        "--manifest", type=str, default="data/manifests/dr_dual_db_manifest.json", help="Path to manifest"
    )
    parser.add_argument("--output-dir", type=str, default="results", help="Results output directory")
    parser.add_argument("--scaler", type=str, default="robust", help="Feature scaler type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    fig_dir = out_root / "figures"
    metrics_dir = out_root / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = Path(args.manifest)
    if not manifest_p.exists():
        logger.info(f"Manifest not found at {manifest_p}. Generating dual-database benchmark automatically...")
        build_diabetic_retinopathy_dual_db(".", samples_per_db=250, seed=args.seed)

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = [DualDatabaseSample.from_dict(d) for d in manifest_data["samples"]]
    class_names = manifest_data["icdr_classes"]
    logger.info(f"Loaded {len(samples)} retinal fundus scans across 2 databases: {manifest_data['databases']}.")

    # 1. Feature Extraction
    logger.info("Extracting 42 clinical biomarker, color, and texture features per scan...")
    t_feat_start = time.perf_counter()
    X, y, feat_names, meta = extract_features_for_dataset(samples)
    t_feat_total = time.perf_counter() - t_feat_start
    logger.info(
        f"Feature matrix extracted: shape={X.shape} in {t_feat_total:.2f}s "
        f"({t_feat_total / len(samples) * 1000:.1f}ms/scan)."
    )

    # 2. Partition Pooled Train/Test Split
    train_indices = [i for i, m in enumerate(meta) if m["split"] == "train"]
    test_indices = [i for i, m in enumerate(meta) if m["split"] == "test"]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    logger.info(f"Dataset split: Train={len(train_indices)} scans, Test={len(test_indices)} scans.")

    # 3. Initialize ML Family Taxonomy Engine & Run All Families
    engine = MLFamilyTaxonomyBenchmarkEngine(class_names=class_names, scaler_type=args.scaler, seed=args.seed)

    logger.info("\n" + "=" * 90)
    logger.info("BENCHMARKING 16 CLASSIFIERS ACROSS 6 MACHINE LEARNING FAMILIES")
    logger.info("=" * 90)

    taxonomy_results = engine.run_all_families_benchmark(X_train, y_train, X_test, y_test)

    # 4. Generate Visualizations
    logger.info("Rendering clinical taxonomy visualization figures...")

    fig_taxonomy = fig_dir / "ml_family_taxonomy_comparison.png"
    plot_ml_taxonomy_grouped_comparison(taxonomy_results, fig_taxonomy)
    logger.info(f"Saved Grouped Family Taxonomy comparison to {fig_taxonomy}")

    fig_radar = fig_dir / "ml_family_radar_comparison.png"
    plot_ml_families_radar_comparison(taxonomy_results, fig_radar)
    logger.info(f"Saved ML Families 5-Axis Radar chart to {fig_radar}")

    fig_best = fig_dir / "ml_family_best_in_family.png"
    plot_best_in_family_comparison(taxonomy_results, fig_best)
    logger.info(f"Saved Best-in-Family Champion comparison to {fig_best}")

    # 5. Export JSON Summary
    summary_export = {
        "benchmark_title": "Diabetic Retinopathy Dual-Database ML Family Taxonomy Benchmark",
        "taxonomy_families": list(ML_TAXONOMY.keys()),
        "total_models": sum(len(f["models"]) for f in taxonomy_results.values()),
        "total_scans": len(samples),
        "feature_count": X.shape[1],
        "feature_extraction_time_sec": round(t_feat_total, 2),
        "families": taxonomy_results,
    }
    json_path = metrics_dir / "dr_ml_family_taxonomy_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    logger.info(f"Saved full taxonomy JSON summary to {json_path}")

    # 6. Formatted Console Presentation
    print("\n" + "=" * 115)
    print("       DIABETIC RETINOPATHY ML FAMILY TAXONOMY BENCHMARK: 16-MODEL CLINICAL LEADERBOARD       ")
    print("=" * 115)
    header = (
        f"{'Family / Algorithm':<36} | {'Accuracy':<9} | {'QWK':<8} | {'Macro F1':<9} | "
        f"{'Spec':<7} | {'Ref. Sens':<10} | {'Latency':<8} | {'Status'}"
    )
    print(header)
    print("-" * 115)

    for f_id, fam in taxonomy_results.items():
        print(f"\n>>> FAMILY: {fam['display_name'].upper()}")
        print(f"    Rationale: {fam['clinical_rationale']}")
        print("-" * 115)
        for m_key, m in fam["models"].items():
            name_disp = m.get("display_name", m_key)[:34]
            acc = f"{m['accuracy'] * 100:.2f}%"
            qwk = f"{m['quadratic_weighted_kappa']:.4f}"
            f1 = f"{m['macro_f1'] * 100:.2f}%"
            spec = f"{m['macro_specificity'] * 100:.2f}%"
            ref_sens = f"{m['binary_referral']['referral_sensitivity'] * 100:.2f}%"
            lat = f"{m['inference_latency_ms']:.2f}ms"
            status = "PASS (>=80%)" if m["quadratic_weighted_kappa"] >= 0.80 else "SUBOPTIMAL"
            is_champ = " [CHAMPION]" if m_key == fam["best_in_family_key"] else ""
            print(
                f"{name_disp + is_champ:<36} | {acc:<9} | {qwk:<8} | {f1:<9} | "
                f"{spec:<7} | {ref_sens:<10} | {lat:<8} | {status}"
            )

    print("\n" + "=" * 115)
    print("                         FAMILY-LEVEL AGGREGATED METRICS COMPARISON                          ")
    print("=" * 115)
    fam_header = (
        f"{'ML Family':<32} | {'Mean QWK':<9} | {'Max QWK':<8} | {'Mean Acc':<9} | "
        f"{'Mean Sens':<10} | {'Mean Spec':<10} | {'Mean Latency'}"
    )
    print(fam_header)
    print("-" * 115)
    for f_id, fam in taxonomy_results.items():
        agg = fam["aggregated"]
        f_name = fam["display_name"][:30]
        m_qwk = f"{agg['mean_qwk']:.4f}"
        mx_qwk = f"{agg['max_qwk']:.4f}"
        m_acc = f"{agg['mean_accuracy'] * 100:.2f}%"
        m_sens = f"{agg['mean_referral_sensitivity'] * 100:.2f}%"
        m_spec = f"{agg['mean_specificity'] * 100:.2f}%"
        m_lat = f"{agg['mean_latency_ms']:.2f}ms"
        print(f"{f_name:<32} | {m_qwk:<9} | {mx_qwk:<8} | {m_acc:<9} | {m_sens:<10} | {m_spec:<10} | {m_lat}")

    print("\n" + "=" * 115)
    print("                             BEST-IN-FAMILY CLINICAL CHAMPIONS                                ")
    print("=" * 115)
    champ_header = (
        f"{'ML Family':<32} | {'Champion Model':<32} | {'QWK':<8} | {'Accuracy':<9} | "
        f"{'Ref. Sens':<10} | {'Latency'}"
    )
    print(champ_header)
    print("-" * 115)
    for f_id, fam in taxonomy_results.items():
        f_name = fam["display_name"][:30]
        champ_name = fam["best_in_family_name"][:30]
        c_qwk = f"{fam['best_in_family_qwk']:.4f}"
        c_acc = f"{fam['best_in_family_accuracy'] * 100:.2f}%"
        c_sens = f"{fam['best_in_family_referral_sens'] * 100:.2f}%"
        champ_res = fam["models"][fam["best_in_family_key"]]
        c_lat = f"{champ_res['inference_latency_ms']:.2f}ms"
        print(f"{f_name:<32} | {champ_name:<32} | {c_qwk:<8} | {c_acc:<9} | {c_sens:<10} | {c_lat}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()

"""
Benchmark Comparison Script: Classical SVM Baseline vs. SOTA Vision Transformer.
Loads evaluation results from both models and produces a comparative leaderboard report
and side-by-side performance summary.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def print_comparison_table(svm_metrics: Optional[Dict[str, Any]], vit_metrics: Optional[Dict[str, Any]]):
    print("\n" + "="*90)
    print("       COLORECTAL HISTOLOGY BENCHMARK COMPARISON: CLASSICAL SVM vs. SOTA ViT")
    print("="*90)
    
    header = f"{'Metric':<32} | {'SVM Baseline (LBP+GLCM+Gabor+Color)':<35} | {'SOTA ViT / EVA-02':<20}"
    print(header)
    print("-"*90)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A (Run main_*.py)"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Top-1 Overall Accuracy", "Accuracy", True),
        ("Balanced Accuracy", "Balanced_Accuracy", True),
        ("Macro Precision", "Macro_Precision", True),
        ("Macro Recall / Sensitivity", "Macro_Recall", True),
        ("Macro F1-Score", "Macro_F1", True),
        ("Weighted F1-Score", "Weighted_F1", True),
        ("Cohen's Kappa (k)", "Cohens_Kappa", False),
        ("Matthews CorrCoef (MCC)", "Matthews_CorrCoef_MCC", False),
        ("Macro ROC-AUC (OvR)", "ROC_AUC_OvR_Macro", True),
        ("Macro PR-AUC (Average)", "PR_AUC_Macro", True),
    ]

    for label, key, is_pct in metrics_to_compare:
        svm_str = format_val(svm_metrics, key, is_pct)
        vit_str = format_val(vit_metrics, key, is_pct)
        print(f"{label:<32} | {svm_str:<35} | {vit_str:<20}")

    print("-"*90)
    print("PER-CLASS F1-SCORE BREAKDOWN:")
    print(f"{'Class Description':<32} | {'SVM F1-Score':<35} | {'ViT F1-Score':<20}")
    print("-"*90)

    if svm_metrics or vit_metrics:
        ref_metrics = vit_metrics if vit_metrics else svm_metrics
        classes = list(ref_metrics.get("Per_Class_Metrics", {}).keys())

        for c in classes:
            s_f1 = svm_metrics.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if svm_metrics else None
            v_f1 = vit_metrics.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if vit_metrics else None

            s_str = f"{s_f1*100:.2f}%" if s_f1 is not None else "N/A"
            v_str = f"{v_f1*100:.2f}%" if v_f1 is not None else "N/A"
            print(f"{c:<32} | {s_str:<35} | {v_str:<20}")

    print("="*90 + "\n")


def plot_side_by_side_bars(svm_metrics: Dict[str, Any], vit_metrics: Dict[str, Any], output_path: Path):
    """
    Generates a comparative bar chart of per-class F1 scores.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(vit_metrics.get("Per_Class_Metrics", {}).keys())
    short_classes = [c.split()[0] for c in classes]

    svm_f1 = [svm_metrics.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]
    vit_f1 = [vit_metrics.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]

    x = np.arange(len(short_classes))
    width = 0.35

    plt.figure(figsize=(12, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width/2, svm_f1, width, label='SVM Baseline (Handcrafted)', color='#4C72B0')
    plt.bar(x + width/2, vit_f1, width, label='Vision Transformer (SOTA)', color='#55A868')

    plt.xlabel('Histological Tissue Classes', fontsize=12, weight='bold')
    plt.ylabel('F1-Score (%)', fontsize=12, weight='bold')
    plt.title('Per-Class F1-Score Comparison: Classical SVM vs. Vision Transformer', fontsize=14, weight='bold')
    plt.xticks(x, short_classes, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] Comparison chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    svm_json = results_dir / "metrics_summary.json"
    vit_json = results_dir / "metrics_summary_vit.json"

    svm_m = load_metrics(svm_json)
    vit_m = load_metrics(vit_json)

    print_comparison_table(svm_m, vit_m)

    if svm_m and vit_m:
        chart_path = results_dir / "baseline_comparison_f1.png"
        plot_side_by_side_bars(svm_m, vit_m, chart_path)
    else:
        if not svm_m:
            print("[Note] SVM metrics missing. Run: python main_svm.py to generate SVM results.")
        if not vit_m:
            print("[Note] ViT metrics missing. Run: python main_vit.py to generate ViT results.")


if __name__ == "__main__":
    main()

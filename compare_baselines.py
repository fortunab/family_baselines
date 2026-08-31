"""
Benchmark Comparison Script:
4-Tier Leaderboard Comparison on Colorectal Histology:
Tier 1: Classical SVM (87.4%)
Tier 2: Modern ConvNeXt CNN (96.3%-97.4%)
Tier 3: SOTA Vision Transformer (98.4%-99.2%)
Tier 4: Computational Pathology Foundation (Virchow / Phikon, >98.5%)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def print_comparison_table(
    svm_m: Optional[Dict[str, Any]],
    cnx_m: Optional[Dict[str, Any]],
    vit_m: Optional[Dict[str, Any]],
    vir_m: Optional[Dict[str, Any]]
):
    print("\n" + "="*125)
    print("           COLORECTAL HISTOLOGY BENCHMARK COMPARISON: 4-TIER LEADERBOARD")
    print("="*125)
    
    header = f"{'Metric':<28} | {'1. Classical SVM':<18} | {'2. ConvNeXt':<16} | {'3. ViT / EVA-02':<18} | {'4. Virchow / Phikon':<20}"
    print(header)
    print("-"*125)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Top-1 Overall Accuracy", "Accuracy", True),
        ("Balanced Accuracy", "Balanced_Accuracy", True),
        ("Macro Precision", "Macro_Precision", True),
        ("Macro Recall / Sens.", "Macro_Recall", True),
        ("Macro F1-Score", "Macro_F1", True),
        ("Weighted F1-Score", "Weighted_F1", True),
        ("Cohen's Kappa (k)", "Cohens_Kappa", False),
        ("Matthews CorrCoef (MCC)", "Matthews_CorrCoef_MCC", False),
        ("Macro ROC-AUC (OvR)", "ROC_AUC_OvR_Macro", True),
        ("Macro PR-AUC (Average)", "PR_AUC_Macro", True),
    ]

    for label, key, is_pct in metrics_to_compare:
        s_str = format_val(svm_m, key, is_pct)
        c_str = format_val(cnx_m, key, is_pct)
        v_str = format_val(vit_m, key, is_pct)
        p_str = format_val(vir_m, key, is_pct)
        print(f"{label:<28} | {s_str:<18} | {c_str:<16} | {v_str:<18} | {p_str:<20}")

    print("-"*125)
    print("PER-CLASS F1-SCORE BREAKDOWN:")
    print(f"{'Class Description':<28} | {'SVM F1':<18} | {'ConvNeXt F1':<16} | {'ViT F1':<18} | {'Virchow F1':<20}")
    print("-"*125)

    available = [m for m in [svm_m, cnx_m, vit_m, vir_m] if m is not None]
    if available:
        ref = available[0]
        classes = list(ref.get("Per_Class_Metrics", {}).keys())

        for c in classes:
            s_f1 = svm_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if svm_m else None
            c_f1 = cnx_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if cnx_m else None
            v_f1 = vit_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if vit_m else None
            p_f1 = vir_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if vir_m else None

            s_str = f"{s_f1*100:.2f}%" if s_f1 is not None else "N/A"
            c_str = f"{c_f1*100:.2f}%" if c_f1 is not None else "N/A"
            v_str = f"{v_f1*100:.2f}%" if v_f1 is not None else "N/A"
            p_str = f"{p_f1*100:.2f}%" if p_f1 is not None else "N/A"
            print(f"{c:<28} | {s_str:<18} | {c_str:<16} | {v_str:<18} | {p_str:<20}")

    print("="*125 + "\n")


def plot_side_by_side_bars(
    svm_m: Optional[Dict[str, Any]],
    cnx_m: Optional[Dict[str, Any]],
    vit_m: Optional[Dict[str, Any]],
    vir_m: Optional[Dict[str, Any]],
    output_path: Path
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available = [m for m in [svm_m, cnx_m, vit_m, vir_m] if m is not None]
    if not available:
        return

    classes = list(available[0].get("Per_Class_Metrics", {}).keys())
    short_classes = [c.split()[0] for c in classes]

    x = np.arange(len(short_classes))
    width = 0.20

    plt.figure(figsize=(15, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    if svm_m:
        svm_f1 = [svm_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]
        plt.bar(x - 1.5*width, svm_f1, width, label='1. Classical SVM (87.4%)', color='#4C72B0')

    if cnx_m:
        cnx_f1 = [cnx_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]
        plt.bar(x - 0.5*width, cnx_f1, width, label='2. ConvNeXt-Tiny (96.3%-97.4%)', color='#E1812C')

    if vit_m:
        vit_f1 = [vit_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]
        plt.bar(x + 0.5*width, vit_f1, width, label='3. Vision Transformer (98.4%-99.2%)', color='#55A868')

    if vir_m:
        vir_f1 = [vir_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score", 0.0)*100 for c in classes]
        plt.bar(x + 1.5*width, vir_f1, width, label='4. Virchow / Phikon (>98.5%)', color='#937860')

    plt.xlabel('Histological Tissue Classes', fontsize=12, weight='bold')
    plt.ylabel('F1-Score (%)', fontsize=12, weight='bold')
    plt.title('Colorectal Histology: 4-Tier Benchmark Comparison', fontsize=14, weight='bold')
    plt.xticks(x, short_classes, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] 4-Tier comparison bar chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    svm_m = load_metrics(results_dir / "metrics_summary.json")
    cnx_m = load_metrics(results_dir / "metrics_summary_convnext.json")
    vit_m = load_metrics(results_dir / "metrics_summary_vit.json")
    vir_m = load_metrics(results_dir / "metrics_summary_virchow.json")

    print_comparison_table(svm_m, cnx_m, vit_m, vir_m)

    chart_path = results_dir / "baseline_comparison_f1.png"
    plot_side_by_side_bars(svm_m, cnx_m, vit_m, vir_m, chart_path)

    # Help tips
    if not svm_m:
        print("[Tip] To run Tier 1 (SVM):        python main_svm.py")
    if not cnx_m:
        print("[Tip] To run Tier 2 (ConvNeXt):   python main_convnext.py")
    if not vit_m:
        print("[Tip] To run Tier 3 (ViT):        python main_vit.py")
    if not vir_m:
        print("[Tip] To run Tier 4 (Virchow):    python main_virchow.py")


if __name__ == "__main__":
    main()

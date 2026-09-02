"""
TensorFlow 4-Foundation Models Comparison Script for Colorectal Histology.
Compares:
1. Meta ConvNeXt-Large (tf.keras.applications.ConvNeXtLarge)
2. Google EfficientNetV2-L (tf.keras.applications.EfficientNetV2L)
3. Google Vision Transformer ViT-Base (google/vit-base-patch16-224-in21k)
4. Google Big Transfer BiT / ResNet152V2 (tf.keras.applications.ResNet152V2)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
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


def print_comparison_table(cnx_m, eff_m, vit_m, bit_m):
    print("\n" + "="*135)
    print("      TENSORFLOW COLORECTAL HISTOLOGY FOUNDATION BENCHMARK: 4-MODEL COMPARISON LEADERBOARD")
    print("="*135)
    
    header = f"{'Metric':<28} | {'1. ConvNeXt-Large':<20} | {'2. EfficientNetV2-L':<20} | {'3. ViT-Base (in21k)':<20} | {'4. BiT / ResNet152V2':<22}"
    print(header)
    print("-"*135)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Top-1 Overall Accuracy", "Accuracy", True),
        ("Balanced Accuracy", "Balanced_Accuracy", True),
        ("Macro Precision", "Macro_Precision", True),
        ("Macro Recall (Sens.)", "Macro_Recall", True),
        ("Macro F1-Score", "Macro_F1", True),
        ("Weighted F1-Score", "Weighted_F1", True),
        ("Cohen's Kappa (k)", "Cohens_Kappa", False),
        ("Matthews CorrCoef (MCC)", "Matthews_CorrCoef_MCC", False),
        ("Macro ROC-AUC (OvR)", "ROC_AUC_OvR_Macro", True)
    ]

    for label, key, is_pct in metrics_to_compare:
        c_str = format_val(cnx_m, key, is_pct)
        e_str = format_val(eff_m, key, is_pct)
        v_str = format_val(vit_m, key, is_pct)
        b_str = format_val(bit_m, key, is_pct)
        print(f"{label:<28} | {c_str:<20} | {e_str:<20} | {v_str:<20} | {b_str:<22}")

    print("-"*135)
    print("PER-CLASS F1-SCORE BREAKDOWN:")
    print(f"{'Tissue Class':<28} | {'ConvNeXt F1':<20} | {'EfficientNet F1':<20} | {'ViT F1':<20} | {'BiT / ResNet F1':<22}")
    print("-"*135)

    available = [m for m in [cnx_m, eff_m, vit_m, bit_m] if m is not None]
    if available:
        ref = available[0]
        classes = list(ref.get("Per_Class_Metrics", {}).keys())

        for c in classes:
            c_f1 = cnx_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if cnx_m else None
            e_f1 = eff_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if eff_m else None
            v_f1 = vit_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if vit_m else None
            b_f1 = bit_m.get("Per_Class_Metrics", {}).get(c, {}).get("F1_Score") if bit_m else None

            c_str = f"{c_f1*100:.2f}%" if c_f1 is not None else "N/A"
            e_str = f"{e_f1*100:.2f}%" if e_f1 is not None else "N/A"
            v_str = f"{v_f1*100:.2f}%" if v_f1 is not None else "N/A"
            b_str = f"{b_f1*100:.2f}%" if b_f1 is not None else "N/A"
            short_c = c.split()[0] if len(c) > 26 else c
            print(f"{short_c:<28} | {c_str:<20} | {e_str:<20} | {v_str:<20} | {b_str:<22}")

    print("-"*135)
    print("EXPERIMENT SEEDS USED:")
    def format_seed(m): return str(m.get("Experiment_Seed", "N/A")) if m else "N/A"
    print(f"{'Active Random Seed':<28} | {format_seed(cnx_m):<20} | {format_seed(eff_m):<20} | {format_seed(vit_m):<20} | {format_seed(bit_m):<22}")
    print("="*135 + "\n")


def plot_side_by_side_bars(cnx_m, eff_m, vit_m, bit_m, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["1. ConvNeXt-L", "2. EfficientNetV2-L", "3. ViT-Base", "4. BiT-ResNet152"]
    metrics_data = [cnx_m, eff_m, vit_m, bit_m]

    accs = [m.get("Accuracy", 0.0)*100 if m else 0.0 for m in metrics_data]
    f1s = [m.get("Macro_F1", 0.0)*100 if m else 0.0 for m in metrics_data]
    aucs = [m.get("ROC_AUC_OvR_Macro", 0.0)*100 if m else 0.0 for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, accs, width, label='Accuracy (%)', color='#2b5c8f')
    plt.bar(x, f1s, width, label='Macro F1 (%)', color='#2ca02c')
    plt.bar(x + width, aucs, width, label='Macro ROC-AUC (%)', color='#d62728')

    plt.xlabel('TensorFlow Foundation Model Architecture', fontsize=11, weight='bold')
    plt.ylabel('Score (%)', fontsize=11, weight='bold')
    plt.title('TensorFlow Colorectal Histology Foundation Leaderboard (70/15/15 Split)', fontsize=13, weight='bold')
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] Leaderboard chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    cnx_m = load_metrics(results_dir / "metrics_summary_convnext_large.json") or load_metrics(results_dir / "metrics_summary_convnext_base.json")
    eff_m = load_metrics(results_dir / "metrics_summary_efficientnetv2_l.json") or load_metrics(results_dir / "metrics_summary_efficientnetv2_m.json")
    vit_m = load_metrics(results_dir / "metrics_summary_vit_base.json")
    bit_m = load_metrics(results_dir / "metrics_summary_bit_resnet152v2.json")

    print_comparison_table(cnx_m, eff_m, vit_m, bit_m)
    plot_side_by_side_bars(cnx_m, eff_m, vit_m, bit_m, results_dir / "tf_foundation_benchmark_comparison.png")


if __name__ == "__main__":
    main()

"""
Comparison & Multi-Center Leaderboard for Foundation Object Detectors on PolypGen2.0.
Compares:
1. Microsoft Florence-2
2. Google OWLv2
3. Grounding DINO
4. Google DeepMind PaliGemma
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


def print_comparison_table(flor_m, owl_m, dino_m, pali_m):
    print("\n" + "="*135)
    print("      POLYPGEN2.0 COCO OBJECT DETECTION BENCHMARK: 4-FOUNDATION MODEL COMPARISON LEADERBOARD")
    print("="*135)

    header = f"{'Metric':<28} | {'1. Florence-2':<22} | {'2. OWLv2':<22} | {'3. Grounding DINO':<22} | {'4. PaliGemma':<22}"
    print(header)
    print("-"*135)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("COCO Primary mAP [50:95]", "mAP_50_95", True),
        ("PASCAL VOC mAP @ 50", "mAP_50", True),
        ("Strict Localization mAP@75", "mAP_75", True),
        ("Mean Bounding Box IoU", "Mean_IoU", False),
        ("Detection Precision @ 50", "Precision_50", True),
        ("Detection Recall @ 50", "Recall_50", True),
        ("Detection F1-Score @ 50", "F1_Score_50", True)
    ]

    for label, key, is_pct in metrics_to_compare:
        f_str = format_val(flor_m, key, is_pct)
        o_str = format_val(owl_m, key, is_pct)
        d_str = format_val(dino_m, key, is_pct)
        p_str = format_val(pali_m, key, is_pct)
        print(f"{label:<28} | {f_str:<22} | {o_str:<22} | {d_str:<22} | {p_str:<22}")

    print("-"*135)
    print("MULTI-CENTER OUT-OF-DISTRIBUTION (OOD) RECALL BREAKDOWN (C1 to C6):")
    print(f"{'Hospital Center':<28} | {'Florence-2 Recall':<22} | {'OWLv2 Recall':<22} | {'DINO Recall':<22} | {'PaliGemma Recall':<22}")
    print("-"*135)

    centers = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for c in centers:
        def get_c_rec(m):
            if m and "Center_Breakdown" in m and c in m["Center_Breakdown"]:
                return f"{m['Center_Breakdown'][c].get('Recall_50', 0.0)*100:.2f}%"
            return "N/A"
        print(f"{'Center ' + c:<28} | {get_c_rec(flor_m):<22} | {get_c_rec(owl_m):<22} | {get_c_rec(dino_m):<22} | {get_c_rec(pali_m):<22}")

    print("-"*135)
    print("EXPERIMENT SEEDS USED:")
    def format_seed(m): return str(m.get("Experiment_Seed", "N/A")) if m else "N/A"
    print(f"{'Active Random Seed':<28} | {format_seed(flor_m):<22} | {format_seed(owl_m):<22} | {format_seed(dino_m):<22} | {format_seed(pali_m):<22}")
    print("="*135 + "\n")


def plot_side_by_side_bars(flor_m, owl_m, dino_m, pali_m, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["1. Florence-2", "2. OWLv2", "3. Grounding DINO", "4. PaliGemma"]
    metrics_data = [flor_m, owl_m, dino_m, pali_m]

    map50s = [m.get("mAP_50", 0.0)*100 if m else 0.0 for m in metrics_data]
    map50_95s = [m.get("mAP_50_95", 0.0)*100 if m else 0.0 for m in metrics_data]
    ious = [m.get("Mean_IoU", 0.0)*100 if m else 0.0 for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, map50s, width, label='mAP @ 50 (%)', color='#1f77b4')
    plt.bar(x, map50_95s, width, label='COCO mAP [50:95] (%)', color='#ff7f0e')
    plt.bar(x + width, ious, width, label='Mean IoU (x100)', color='#2ca02c')

    plt.xlabel('Foundation Object Detection Architecture', fontsize=11, weight='bold')
    plt.ylabel('Benchmark Score (%)', fontsize=11, weight='bold')
    plt.title('PolypGen2.0 Hugging Face COCO Object Detection Leaderboard', fontsize=13, weight='bold')
    plt.xticks(x, models, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] Detection leaderboard chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    flor_m = load_metrics(results_dir / "metrics_summary_florence2.json")
    owl_m = load_metrics(results_dir / "metrics_summary_owlv2.json")
    dino_m = load_metrics(results_dir / "metrics_summary_grounding_dino.json")
    pali_m = load_metrics(results_dir / "metrics_summary_paligemma.json")

    print_comparison_table(flor_m, owl_m, dino_m, pali_m)
    plot_side_by_side_bars(flor_m, owl_m, dino_m, pali_m, results_dir / "polypgen_detection_benchmark_comparison.png")


if __name__ == "__main__":
    main()

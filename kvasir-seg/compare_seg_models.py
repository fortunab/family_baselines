"""
Comparison & Leaderboard Script for Foundation Segmentation Models on Kvasir-SEG.
Compares:
1. MedSAM / Segment Anything Model
2. NVIDIA SegFormer-B3
3. Meta ConvNeXt-Base U-Net
4. Microsoft Swin-Base U-Net
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


def print_comparison_table(med_m, seg_m, cnx_m, swin_m):
    print("\n" + "="*135)
    print("      KVASIR-SEG FOUNDATION SEMANTIC SEGMENTATION BENCHMARK: 4-MODEL COMPARISON LEADERBOARD")
    print("="*135)

    header = f"{'Metric':<32} | {'1. MedSAM (ViT-B)':<22} | {'2. SegFormer-B3':<22} | {'3. ConvNeXt-UNet':<22} | {'4. Swin-UNet':<22}"
    print(header)
    print("-"*135)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Dice Similarity Coeff (DSC)", "Dice_Coefficient", True),
        ("Mean IoU (Jaccard Index)", "mIoU_Jaccard", True),
        ("Precision (Positive Pred)", "Precision", True),
        ("Recall (Lesion Sensitivity)", "Recall_Sensitivity", True),
        ("Specificity (True Negative)", "Specificity", True),
        ("Pixel Accuracy (PA)", "Pixel_Accuracy", True)
    ]

    for label, key, is_pct in metrics_to_compare:
        m_str = format_val(med_m, key, is_pct)
        s_str = format_val(seg_m, key, is_pct)
        c_str = format_val(cnx_m, key, is_pct)
        w_str = format_val(swin_m, key, is_pct)
        print(f"{label:<32} | {m_str:<22} | {s_str:<22} | {c_str:<22} | {w_str:<22}")

    print("-"*135)
    print("EXPERIMENT SEEDS USED:")
    def format_seed(m): return str(m.get("Experiment_Seed", "N/A")) if m else "N/A"
    print(f"{'Active Random Seed':<32} | {format_seed(med_m):<22} | {format_seed(seg_m):<22} | {format_seed(cnx_m):<22} | {format_seed(swin_m):<22}")
    print("="*135 + "\n")


def plot_side_by_side_bars(med_m, seg_m, cnx_m, swin_m, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["1. MedSAM", "2. SegFormer-B3", "3. ConvNeXt-UNet", "4. Swin-UNet"]
    metrics_data = [med_m, seg_m, cnx_m, swin_m]

    dices = [m.get("Dice_Coefficient", 0.0)*100 if m else 0.0 for m in metrics_data]
    mious = [m.get("mIoU_Jaccard", 0.0)*100 if m else 0.0 for m in metrics_data]
    recalls = [m.get("Recall_Sensitivity", 0.0)*100 if m else 0.0 for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, dices, width, label='Dice Score / DSC (%)', color='#2b5c8f')
    plt.bar(x, mious, width, label='Mean IoU / Jaccard (%)', color='#2ca02c')
    plt.bar(x + width, recalls, width, label='Sensitivity / Recall (%)', color='#d62728')

    plt.xlabel('Foundation Segmentation Model Architecture', fontsize=11, weight='bold')
    plt.ylabel('Clinical Score (%)', fontsize=11, weight='bold')
    plt.title('Kvasir-SEG Foundation Semantic Segmentation Leaderboard', fontsize=13, weight='bold')
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

    med_m = load_metrics(results_dir / "metrics_summary_medsam.json")
    seg_m = load_metrics(results_dir / "metrics_summary_segformer.json")
    cnx_m = load_metrics(results_dir / "metrics_summary_convnext_unet.json")
    swin_m = load_metrics(results_dir / "metrics_summary_swin_unet.json")

    print_comparison_table(med_m, seg_m, cnx_m, swin_m)
    plot_side_by_side_bars(med_m, seg_m, cnx_m, swin_m, results_dir / "kvasir_seg_benchmark_comparison.png")


if __name__ == "__main__":
    main()

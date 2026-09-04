"""
Comparison Leaderboard & Visualization Generator for fastai & skorch Colorectal Histology Suite.
Compares:
1. fastai ConvNeXt-Base
2. fastai Vision Transformer (ViT-Base)
3. fastai EfficientNetV2
4. skorch ResNet50
"""

import os
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


def print_comparison_table(models_data: List[Dict[str, Any]]):
    print("\n" + "="*140)
    print("      COLORECTAL HISTOLOGY BENCHMARK: FASTAI & SKORCH MULTI-MODEL LEADERBOARD")
    print("="*140)

    header = f"{'Metric':<28}"
    for m in models_data:
        name = f"{m.get('name', 'Model')}"
        header += f" | {name:<24}"
    print(header)
    print("-"*140)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Test Accuracy", "Accuracy", True),
        ("Balanced Accuracy", "Balanced_Accuracy", True),
        ("Macro Precision", "Macro_Precision", True),
        ("Macro Recall", "Macro_Recall", True),
        ("Macro F1-Score", "Macro_F1", True),
        ("Weighted F1-Score", "Weighted_F1", True),
        ("Multi-Class ROC-AUC", "ROC_AUC_Macro", False)
    ]

    for label, key, is_pct in metrics_to_compare:
        row = f"{label:<28}"
        for m in models_data:
            val_str = format_val(m.get("metrics"), key, is_pct)
            row += f" | {val_str:<24}"
        print(row)

    print("-"*140)
    print("EXPERIMENT METADATA:")
    row_fw = f"{'Framework':<28}"
    row_bb = f"{'Backbone':<28}"
    for m in models_data:
        met = m.get("metrics") or {}
        row_fw += f" | {met.get('Framework', 'N/A'):<24}"
        row_bb += f" | {met.get('Backbone', 'N/A'):<24}"
    print(row_fw)
    print(row_bb)
    print("="*140 + "\n")


def plot_side_by_side_bars(models_data: List[Dict[str, Any]], output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [m.get("name", "Model") for m in models_data]
    accs = [m.get("metrics", {}).get("Accuracy", 0.0)*100 if m.get("metrics") else 0.0 for m in models_data]
    bal_accs = [m.get("metrics", {}).get("Balanced_Accuracy", 0.0)*100 if m.get("metrics") else 0.0 for m in models_data]
    f1s = [m.get("metrics", {}).get("Macro_F1", 0.0)*100 if m.get("metrics") else 0.0 for m in models_data]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, accs, width, label='Test Accuracy (%)', color='#2b5c8f')
    plt.bar(x, bal_accs, width, label='Balanced Accuracy (%)', color='#2ca02c')
    plt.bar(x + width, f1s, width, label='Macro F1-Score (%)', color='#e67e22')

    plt.xlabel('Foundation Model Architecture & Framework', fontsize=11, weight='bold')
    plt.ylabel('Performance Metric (%)', fontsize=11, weight='bold')
    plt.title('Colorectal Histology Benchmark Leaderboard (fastai & skorch)', fontsize=13, weight='bold')
    plt.xticks(x, names, fontsize=10)
    plt.ylim(0, 105)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Compare] Leaderboard comparison chart saved to: {output_path}")


def main():
    project_root = Path(__file__).parent.resolve()
    results_dir = project_root / "results"

    candidates = [
        {"name": "fastai ConvNeXt", "file": "metrics_fastai_convnext_base.json"},
        {"name": "fastai ViT-Base", "file": "metrics_fastai_vit_base_patch16_224.json"},
        {"name": "fastai EfficientNet", "file": "metrics_fastai_efficientnet_b3.json"},
        {"name": "skorch ResNet50", "file": "metrics_skorch_resnet50d.json"}
    ]

    models_data = []
    for c in candidates:
        m = load_metrics(results_dir / c["file"])
        models_data.append({"name": c["name"], "metrics": m})

    print_comparison_table(models_data)
    plot_side_by_side_bars(models_data, results_dir / "colorectal_histology_fastai_comparison.png")


if __name__ == "__main__":
    main()

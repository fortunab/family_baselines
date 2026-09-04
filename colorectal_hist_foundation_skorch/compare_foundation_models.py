"""
Leaderboard Comparison & Visualization Script for skorch Pathology Foundation Models Suite.
Compares:
1. Owkin Phikon (owkin/phikon)
2. Paige Virchow (paige-ai/Virchow)
3. Harvard UNI (MahmoodLab/UNI)
4. Meta DINOv2 (facebook/dinov2-base)
5. Microsoft BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_metrics(json_path: Path) -> Optional[Dict[str, Any]]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def print_comparison_table(models_data: List[Dict[str, Any]]):
    print("\n" + "=" * 145)
    print("    SKORCH PATHOLOGY FOUNDATION BENCHMARK LEADERBOARD: COLORECTAL HISTOLOGY")
    print("=" * 145)

    header = f"{'Metric':<28}"
    for m in models_data:
        name = f"{m.get('name', 'Model')}"
        header += f" | {name:<20}"
    print(header)
    print("-" * 145)

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
        ("Multi-Class ROC-AUC", "ROC_AUC_Macro", False),
    ]

    for label, key, is_pct in metrics_to_compare:
        row = f"{label:<28}"
        for m in models_data:
            val_str = format_val(m.get("metrics"), key, is_pct)
            row += f" | {val_str:<20}"
        print(row)

    print("-" * 145)
    print("CONFIG PROFILES:")
    row_cfg = f"{'TOML Config':<28}"
    for m in models_data:
        row_cfg += f" | {m.get('config_file', 'N/A'):<20}"
    print(row_cfg)
    print("=" * 145 + "\n")


def plot_side_by_side_bars(models_data: List[Dict[str, Any]], output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [m.get("short_name", "Model") for m in models_data]
    accs = [
        m.get("metrics", {}).get("Accuracy", 0.0) * 100 if m.get("metrics") else 0.0
        for m in models_data
    ]
    bal_accs = [
        m.get("metrics", {}).get("Balanced_Accuracy", 0.0) * 100 if m.get("metrics") else 0.0
        for m in models_data
    ]
    f1s = [
        m.get("metrics", {}).get("Macro_F1", 0.0) * 100 if m.get("metrics") else 0.0
        for m in models_data
    ]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(12, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, accs, width, label="Test Accuracy (%)", color="#2b5c8f")
    plt.bar(x, bal_accs, width, label="Balanced Accuracy (%)", color="#2ca02c")
    plt.bar(x + width, f1s, width, label="Macro F1-Score (%)", color="#e67e22")

    plt.xlabel("Pathology Foundation Architecture", fontsize=11, weight="bold")
    plt.ylabel("Clinical Performance (%)", fontsize=11, weight="bold")
    plt.title(
        "Colorectal Histology 8-Class Benchmark Leaderboard (skorch Foundation Models)",
        fontsize=13,
        weight="bold",
    )
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
        {
            "name": "Owkin Phikon",
            "short_name": "Phikon",
            "file": "metrics_owkin_phikon.json",
            "config_file": "phikon.toml",
        },
        {
            "name": "Paige Virchow",
            "short_name": "Virchow",
            "file": "metrics_paige_ai_Virchow.json",
            "config_file": "virchow.toml",
        },
        {
            "name": "Harvard UNI",
            "short_name": "UNI",
            "file": "metrics_MahmoodLab_UNI.json",
            "config_file": "uni.toml",
        },
        {
            "name": "Meta DINOv2",
            "short_name": "DINOv2",
            "file": "metrics_facebook_dinov2_base.json",
            "config_file": "dinov2.toml",
        },
        {
            "name": "BiomedCLIP",
            "short_name": "BiomedCLIP",
            "file": "metrics_microsoft_BiomedCLIP_PubMedBERT_256_vit_base_patch16_224.json",
            "config_file": "biomedclip.toml",
        },
    ]

    models_data = []
    for c in candidates:
        m = load_metrics(results_dir / c["file"])
        models_data.append({**c, "metrics": m})

    print_comparison_table(models_data)
    plot_side_by_side_bars(models_data, results_dir / "pathology_foundation_skorch_comparison.png")


if __name__ == "__main__":
    main()

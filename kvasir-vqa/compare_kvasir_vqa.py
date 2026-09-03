"""
Comparison & Leaderboard Script for Foundation VQA Models on SimulaMet-HOST/Kvasir-VQA.
Compares:
1. Microsoft Florence-2
2. Google DeepMind PaliGemma
3. Alibaba Qwen2-VL
4. Salesforce BLIP-2
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


def print_comparison_table(flor_m, pali_m, qwen_m, blip_m):
    print("\n" + "="*135)
    print("      KVASIR-VQA FOUNDATION MODEL BENCHMARK: 4-MODEL COMPARISON LEADERBOARD")
    print("="*135)

    header = f"{'Metric':<32} | {'1. Florence-2':<22} | {'2. PaliGemma':<22} | {'3. Qwen2-VL':<22} | {'4. BLIP-2':<22}"
    print(header)
    print("-"*135)

    def format_val(metrics, key, pct=True):
        if metrics is None or key not in metrics:
            return "N/A"
        val = metrics[key]
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    metrics_to_compare = [
        ("Overall VQA Accuracy (EM)", "Overall_Accuracy_EM", True),
        ("Closed-Ended (Yes/No) Acc", "Yes_No_Accuracy", True),
        ("Closed-Ended (Yes/No) F1", "Yes_No_F1", True),
        ("Linguistic BLEU-1 Score", "BLEU_1", True),
        ("Linguistic BLEU-4 Score", "BLEU_4", True),
        ("Linguistic ROUGE-L F1", "ROUGE_L_F1", True),
        ("Semantic Token-Level F1", "Token_F1", True)
    ]

    for label, key, is_pct in metrics_to_compare:
        f_str = format_val(flor_m, key, is_pct)
        p_str = format_val(pali_m, key, is_pct)
        q_str = format_val(qwen_m, key, is_pct)
        b_str = format_val(blip_m, key, is_pct)
        print(f"{label:<32} | {f_str:<22} | {p_str:<22} | {q_str:<22} | {b_str:<22}")

    print("-"*135)
    print("EXPERIMENT SEEDS USED:")
    def format_seed(m): return str(m.get("Experiment_Seed", "N/A")) if m else "N/A"
    print(f"{'Active Random Seed':<32} | {format_seed(flor_m):<22} | {format_seed(pali_m):<22} | {format_seed(qwen_m):<22} | {format_seed(blip_m):<22}")
    print("="*135 + "\n")


def plot_side_by_side_bars(flor_m, pali_m, qwen_m, blip_m, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = ["1. Florence-2", "2. PaliGemma", "3. Qwen2-VL", "4. BLIP-2"]
    metrics_data = [flor_m, pali_m, qwen_m, blip_m]

    accs = [m.get("Overall_Accuracy_EM", 0.0)*100 if m else 0.0 for m in metrics_data]
    yn_accs = [m.get("Yes_No_Accuracy", 0.0)*100 if m else 0.0 for m in metrics_data]
    rouges = [m.get("ROUGE_L_F1", 0.0)*100 if m else 0.0 for m in metrics_data]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    plt.bar(x - width, accs, width, label='Overall VQA Accuracy (EM %)', color='#2b5c8f')
    plt.bar(x, yn_accs, width, label='Yes/No Accuracy (%)', color='#2ca02c')
    plt.bar(x + width, rouges, width, label='ROUGE-L F1 Score (%)', color='#e67e22')

    plt.xlabel('Endoscopy Foundation Model Architecture', fontsize=11, weight='bold')
    plt.ylabel('Clinical Score (%)', fontsize=11, weight='bold')
    plt.title('Kvasir-VQA Gastrointestinal Foundation Benchmark Leaderboard', fontsize=13, weight='bold')
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

    flor_m = load_metrics(results_dir / "metrics_summary_florence2.json")
    pali_m = load_metrics(results_dir / "metrics_summary_paligemma.json")
    qwen_m = load_metrics(results_dir / "metrics_summary_qwen2vl.json")
    blip_m = load_metrics(results_dir / "metrics_summary_blip2.json")

    print_comparison_table(flor_m, pali_m, qwen_m, blip_m)
    plot_side_by_side_bars(flor_m, pali_m, qwen_m, blip_m, results_dir / "kvasir_vqa_benchmark_comparison.png")


if __name__ == "__main__":
    main()

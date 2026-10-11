"""Visualization Engine for Alzheimer MRI Grayscale Benchmark.

Generates:
1. Grayscale Preprocessing Pipeline Breakdown (Raw, Grayscale, Denoised, CLAHE, Mask).
2. ML Family Performance Bar Chart (with 80% target threshold line).
3. ML Family vs Foundational Models Comparative Leaderboard Matrix.
4. Confusion Matrix Heatmaps.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_preprocessing_breakdown(
    preproc_results: Dict[str, np.ndarray],
    class_name: str,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Renders a 5-panel diagnostic breakdown of the grayscale preprocessing pipeline."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor("#0D1117")

    titles = [
        "1. Raw MRI Slice",
        "2. Grayscale Conversion",
        "3. Noise Filtration",
        "4. CLAHE Contrast Enhanced",
        "5. Parenchymal Masked Output",
    ]
    keys = ["raw_gray", "raw_gray", "denoised", "clahe_enhanced", "processed"]

    for idx, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        ax.set_facecolor("#161B22")
        img = preproc_results.get(key, np.zeros((128, 128)))
        ax.imshow(img, cmap="gray")
        ax.set_title(title, color="#58A6FF", fontsize=10, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(
        f"Alzheimer MRI Neuroimaging Grayscale Preprocessing Pipeline ({class_name})",
        fontsize=14,
        color="#F0F6FC",
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_ml_family_performance(
    ml_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Renders grouped bar chart of ML Family performance with >80% threshold line."""
    model_names = list(ml_metrics.keys())
    metric_keys = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Bal. Acc", "Precision", "Recall", "F1 Macro"]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    x = np.arange(len(model_names))
    width = 0.15
    colors = ["#58A6FF", "#3FB950", "#D29922", "#F0883E", "#BC8CFF"]

    for idx, (m_key, m_lbl) in enumerate(zip(metric_keys, metric_labels)):
        vals = [ml_metrics[m].get(m_key, 0.0) for m in model_names]
        ax.bar(x + idx * width - 2 * width, vals, width, label=m_lbl, color=colors[idx])

    # Draw 80% threshold line
    ax.axhline(0.80, color="#FF7B72", linestyle="--", linewidth=2.0, label="Target Threshold (>80%)")
    ax.text(
        len(model_names) - 0.5,
        0.81,
        "Target Threshold: 80%",
        color="#FF7B72",
        fontsize=10,
        fontweight="bold",
        ha="right",
    )

    ax.set_title(
        "Classical ML Family Classification Performance (Grayscaled Alzheimer MRI)",
        color="#F0F6FC",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color="#C9D1D9", fontsize=10, fontweight="bold", rotation=15)
    ax.tick_params(colors="#8B949E")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#C9D1D9", loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#8B949E")

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_ml_vs_foundation_benchmark(
    ml_metrics: Dict[str, Dict[str, Any]],
    foundation_metrics: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Renders comprehensive comparative matrix: Classical ML Family vs Deep Vision Foundation Models."""
    all_models = list(ml_metrics.keys()) + list(foundation_metrics.keys())
    categories = ["ML Family"] * len(ml_metrics) + ["Foundation"] * len(foundation_metrics)

    accs = [ml_metrics[m].get("accuracy", 0.0) for m in ml_metrics] + [
        foundation_metrics[m].get("accuracy", 0.0) for m in foundation_metrics
    ]
    f1s = [ml_metrics[m].get("f1_macro", 0.0) for m in ml_metrics] + [
        foundation_metrics[m].get("f1_macro", 0.0) for m in foundation_metrics
    ]
    latencies = [ml_metrics[m].get("inference_latency_ms_per_slice", 0.0) for m in ml_metrics] + [
        foundation_metrics[m].get("inference_latency_ms_per_slice", 0.0) for m in foundation_metrics
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0D1117")

    # Panel 1: Accuracy & F1 Macro
    ax1.set_facecolor("#161B22")
    x = np.arange(len(all_models))
    width = 0.38

    ax1.bar(x - width / 2, accs, width, label="Accuracy", color="#58A6FF")
    ax1.bar(x + width / 2, f1s, width, label="F1 Macro", color="#3FB950")
    ax1.axhline(0.80, color="#FF7B72", linestyle="--", linewidth=1.8, label="Target (>80%)")

    ax1.set_title("Diagnostic Accuracy & Macro F1 (Target > 80%)", color="#F0F6FC", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_models, rotation=35, ha="right", color="#C9D1D9", fontsize=9)
    ax1.tick_params(colors="#8B949E")
    ax1.set_ylim(0, 1.05)
    ax1.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#C9D1D9")
    ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#8B949E")

    # Panel 2: Inference Latency
    ax2.set_facecolor("#161B22")
    c_bars = ["#A371F7" if cat == "ML Family" else "#F0883E" for cat in categories]
    bars = ax2.bar(x, latencies, color=c_bars, width=0.55)

    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{h:.1f}ms",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#C9D1D9",
            fontweight="bold",
        )

    ax2.set_title("Inference Latency (ms per slice)", color="#F0F6FC", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_models, rotation=35, ha="right", color="#C9D1D9", fontsize=9)
    ax2.tick_params(colors="#8B949E")
    ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#8B949E")

    fig.suptitle(
        "Grand Benchmark: Classical ML Family vs Vision Foundation Models (Grayscale MRI)",
        fontsize=14,
        color="#F0F6FC",
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_multiscale_patch_grid_comparison(
    gray_128: np.ndarray,
    img_32: np.ndarray,
    img_8: np.ndarray,
    saliency_32: Optional[np.ndarray] = None,
    saliency_8: Optional[np.ndarray] = None,
    class_name: str = "Alzheimer MRI",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Renders a 6-panel diagnostic comparison of Multi-Scale Downscaling (32x32 and 8x8) and Patch Grids."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0D1117")

    fig.suptitle(
        f"Multi-Resolution Patch-Based Neuroimaging Analysis | {class_name}",
        fontsize=15,
        color="#F0F6FC",
        fontweight="bold",
        y=0.98,
    )

    # 1. Base Grayscale (128x128)
    axes[0, 0].set_facecolor("#161B22")
    axes[0, 0].imshow(gray_128, cmap="gray")
    axes[0, 0].set_title("1. Preprocessed Slice (128x128)", color="#58A6FF", fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Downscaled 32x32 with 4x4 Patch Grid overlay
    axes[0, 1].set_facecolor("#161B22")
    axes[0, 1].imshow(img_32, cmap="gray")
    # Draw patch grid lines for 8x8 patches (4x4 grid)
    for line_idx in range(1, 4):
        axes[0, 1].axhline(line_idx * 8 - 0.5, color="#FF7B72", linewidth=1.5, linestyle="--")
        axes[0, 1].axvline(line_idx * 8 - 0.5, color="#FF7B72", linewidth=1.5, linestyle="--")
    axes[0, 1].set_title("2. Downscaled 32x32 (16 Patches of 8x8)", color="#3FB950", fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Patch Tiling Matrix
    axes[0, 2].set_facecolor("#161B22")
    # Render an extracted 4x4 mosaic of the 16 patches with spacing
    mosaic = np.zeros((36, 36), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            p = img_32[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
            mosaic[r * 9 : r * 9 + 8, c * 9 : c * 9 + 8] = p
    axes[0, 2].imshow(mosaic, cmap="gray")
    axes[0, 2].set_title("3. Disjoint Patch Tiling (Patches Only)", color="#D29922", fontweight="bold")
    axes[0, 2].axis("off")

    # 4. Downscaled 8x8 Macro-Patch Grid
    axes[1, 0].set_facecolor("#161B22")
    axes[1, 0].imshow(img_8, cmap="gray", interpolation="nearest")
    # Quadrant grid lines
    axes[1, 0].axhline(3.5, color="#A371F7", linewidth=2.0, linestyle=":")
    axes[1, 0].axvline(3.5, color="#A371F7", linewidth=2.0, linestyle=":")
    axes[1, 0].set_title("4. Downscaled 8x8 (Macro-Patches & Quadrants)", color="#A371F7", fontweight="bold")
    axes[1, 0].axis("off")

    # 5. 32x32 Patch Saliency Heatmap
    axes[1, 1].set_facecolor("#161B22")
    if saliency_32 is not None:
        im5 = axes[1, 1].imshow(saliency_32, cmap="inferno", interpolation="nearest")
        for r in range(4):
            for c in range(4):
                val = saliency_32[r, c]
                axes[1, 1].text(
                    c, r, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold"
                )
        cbar5 = fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(colors="#8B949E")
    axes[1, 1].set_title("5. 32x32 Patch Importance Saliency", color="#F0883E", fontweight="bold")
    axes[1, 1].axis("off")

    # 6. 8x8 Macro Saliency Map
    axes[1, 2].set_facecolor("#161B22")
    if saliency_8 is not None:
        im6 = axes[1, 2].imshow(saliency_8, cmap="plasma", interpolation="nearest")
        cbar6 = fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(colors="#8B949E")
    else:
        axes[1, 2].imshow(img_8, cmap="plasma", interpolation="nearest")
    axes[1, 2].set_title("6. 8x8 Macro-Patch Saliency Map", color="#BC8CFF", fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig


def plot_patch_resolution_comparison(
    res_benchmarks: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots classification performance across patch resolutions (32x32 vs 8x8 vs Multi-Scale)."""
    scales = list(res_benchmarks.keys())
    accs = [res_benchmarks[s].get("accuracy", 0.0) for s in scales]
    f1s = [res_benchmarks[s].get("f1_macro", 0.0) for s in scales]
    precs = [res_benchmarks[s].get("precision_macro", 0.0) for s in scales]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    x = np.arange(len(scales))
    width = 0.25

    ax.bar(x - width, accs, width, label="Accuracy", color="#58A6FF")
    ax.bar(x, f1s, width, label="F1 Macro", color="#3FB950")
    ax.bar(x + width, precs, width, label="Precision Macro", color="#D29922")

    ax.axhline(0.80, color="#FF7B72", linestyle="--", linewidth=2.0, label="Target (>80%)")
    ax.text(len(scales) - 0.5, 0.81, "Target: 80%", color="#FF7B72", fontweight="bold", ha="right")

    ax.set_title("Patch-Only Classification Across Downscaled Resolutions", color="#F0F6FC", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, color="#C9D1D9", fontweight="bold")
    ax.tick_params(colors="#8B949E")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#21262D", edgecolor="#30363D", labelcolor="#C9D1D9")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#8B949E")

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return fig

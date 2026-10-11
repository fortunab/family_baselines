"""Publication-Quality Visualizations for Multi-Scale Downscaling Preprocessing.

Generates:
1. Multi-Scale Anatomical Progression Gallery (Original -> 32x32 -> 8x8 + Reconstructions & Error Maps)
2. Information Fidelity and Compression Tradeoff Analysis (Entropy, SSIM, PSNR, Edge Energy)
3. Machine Learning Classification Performance: 32x32 vs 8x8 Benchmark Comparison
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Use seaborn whitegrid / clean dark theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_multiscale_progression_gallery(
    class_samples: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders a 4-row x 5-column grid showing anatomical degradation and reconstruction fidelity.

    Columns:
    1: Original MRI Slice (176x208)
    2: Downscaled 32x32 Grayscale
    3: 32x32 Reconstruction & Absolute Error
    4: Downscaled 8x8 Grayscale (64-pixel macro grid)
    5: 8x8 Reconstruction & Absolute Error
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(class_samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.8 * n_rows), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (cls_name, data) in enumerate(class_samples.items()):
        orig_gray = data["orig_gray"]
        arr_32 = data["arr_32"]
        arr_8 = data["arr_8"]
        recon_32 = data["recon_32"]
        recon_8 = data["recon_8"]

        error_32 = np.abs(orig_gray - recon_32)
        error_8 = np.abs(orig_gray - recon_8)

        # Style each axis in row
        for c in range(5):
            ax = axes[row_idx, c]
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.axis("off")

        # Col 0: Original
        ax0 = axes[row_idx, 0]
        ax0.imshow(orig_gray, cmap="bone")
        ax0.set_title(
            f"{cls_name}\nOriginal (176x208)",
            color="#f8fafc",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        # Col 1: Downscaled 32x32
        ax1 = axes[row_idx, 1]
        ax1.imshow(arr_32, cmap="bone", interpolation="nearest")
        ax1.set_title(
            "Context 1: 32x32\n(1,024 px, 35.8x compr.)",
            color="#38bdf8",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        # Col 2: 32x32 Reconstruction Error
        ax2 = axes[row_idx, 2]
        ax2.imshow(error_32, cmap="magma")
        ax2.set_title(
            f"32x32 Recon. Error\nSSIM: {data.get('ssim_32', 0.0):.3f} | PSNR: {data.get('psnr_32', 0.0):.1f}dB",
            color="#a78bfa",
            fontsize=10.5,
            pad=6,
        )

        # Col 3: Downscaled 8x8
        ax3 = axes[row_idx, 3]
        ax3.imshow(arr_8, cmap="bone", interpolation="nearest")
        ax3.set_title(
            "Context 2: 8x8\n(64 px, 572.0x compr.)",
            color="#fbbf24",
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        # Col 4: 8x8 Reconstruction Error
        ax4 = axes[row_idx, 4]
        ax4.imshow(error_8, cmap="magma")
        ax4.set_title(
            f"8x8 Recon. Error\nSSIM: {data.get('ssim_8', 0.0):.3f} | PSNR: {data.get('psnr_8', 0.0):.1f}dB",
            color="#f43f5e",
            fontsize=10.5,
            pad=6,
        )

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_information_fidelity_analysis(
    df_manifest: pd.DataFrame,
    output_path: Union[str, Path],
):
    """Renders 4-panel analysis of Shannon entropy, SSIM, PSNR, and edge energy preservation."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color("#334155")

    # 1. Shannon Entropy Preservation (32x32 vs 8x8)
    ax1 = axes[0, 0]
    classes = sorted(df_manifest["class_name"].unique())
    x = np.arange(len(classes))
    width = 0.35

    mean_ent_32 = [df_manifest[df_manifest["class_name"] == c]["entropy_32x32"].mean() for c in classes]
    mean_ent_8 = [df_manifest[df_manifest["class_name"] == c]["entropy_8x8"].mean() for c in classes]

    b1 = ax1.bar(x - width / 2, mean_ent_32, width, label="32x32 Grayscale", color="#38bdf8", edgecolor="#0284c7")
    b2 = ax1.bar(x + width / 2, mean_ent_8, width, label="8x8 Grayscale", color="#fbbf24", edgecolor="#d97706")

    ax1.set_ylabel("Shannon Entropy (bits)", color="#f8fafc", fontsize=11)
    ax1.set_title(
        "Information Entropy Retention Across Stages", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, color="#f8fafc", fontsize=10, rotation=15)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    for bar in b1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            color="#38bdf8",
            fontsize=8.5,
        )
    for bar in b2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.05,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            color="#fbbf24",
            fontsize=8.5,
        )

    # 2. SSIM Reconstruction Quality
    ax2 = axes[0, 1]
    df_ssim = pd.melt(
        df_manifest,
        id_vars=["class_name"],
        value_vars=["ssim_recon_32x32", "ssim_recon_8x8"],
        var_name="Resolution",
        value_name="SSIM",
    )
    df_ssim["Resolution"] = df_ssim["Resolution"].replace(
        {
            "ssim_recon_32x32": "32x32 (1024 px)",
            "ssim_recon_8x8": "8x8 (64 px)",
        }
    )
    sns.boxplot(data=df_ssim, x="class_name", y="SSIM", hue="Resolution", ax=ax2, palette=["#38bdf8", "#fbbf24"])
    ax2.set_title(
        "Structural Similarity Index (SSIM) vs Original", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10
    )
    ax2.set_ylabel("SSIM", color="#f8fafc")
    ax2.set_xlabel("")
    ax2.tick_params(colors="#f8fafc")
    ax2.tick_params(axis="x", rotation=15)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    # 3. PSNR Reconstruction Fidelity (dB)
    ax3 = axes[1, 0]
    df_psnr = pd.melt(
        df_manifest,
        id_vars=["class_name"],
        value_vars=["psnr_recon_32x32", "psnr_recon_8x8"],
        var_name="Resolution",
        value_name="PSNR_dB",
    )
    df_psnr["Resolution"] = df_psnr["Resolution"].replace(
        {
            "psnr_recon_32x32": "32x32 (1024 px)",
            "psnr_recon_8x8": "8x8 (64 px)",
        }
    )
    sns.boxplot(data=df_psnr, x="class_name", y="PSNR_dB", hue="Resolution", ax=ax3, palette=["#38bdf8", "#fbbf24"])
    ax3.set_title("Peak Signal-to-Noise Ratio (PSNR in dB)", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10)
    ax3.set_ylabel("PSNR (dB)", color="#f8fafc")
    ax3.set_xlabel("")
    ax3.tick_params(colors="#f8fafc")
    ax3.tick_params(axis="x", rotation=15)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    # 4. Dimensionality & Compression Ratio Tradeoff
    ax4 = axes[1, 1]
    levels = ["Original Raw", "Context 1: 32x32", "Context 2: 8x8"]
    pixels = [36608, 1024, 64]
    comp_factors = [1.0, 36608 / 1024, 36608 / 64]
    colors = ["#94a3b8", "#38bdf8", "#fbbf24"]

    y_pos = np.arange(len(levels))
    bars = ax4.barh(y_pos, comp_factors, color=colors, edgecolor="#334155")
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(levels, color="#f8fafc", fontsize=10.5)
    ax4.set_xlabel("Compression Factor (Multiplier relative to 176x208)", color="#f8fafc", fontsize=11)
    ax4.set_title(
        "Compression Ratio: 32x32 vs 8x8 Downscaling", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=10
    )
    ax4.set_xlim([0, 650])

    for bar, px, fac in zip(bars, pixels, comp_factors):
        w = bar.get_width()
        ax4.text(
            w + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{fac:.1f}x ({px} px)",
            va="center",
            color="#f8fafc",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_ml_benchmark_comparison_32_vs_8(
    eval_32: Dict[str, Dict[str, float]],
    eval_8: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
):
    """Renders grouped bar chart comparing ML family classification performance between 32x32 and 8x8."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    models = sorted(list(eval_32.keys()))
    x = np.arange(len(models))
    width = 0.35

    # Panel 1: Accuracy & Balanced Accuracy
    ax1 = axes[0]
    acc_32 = [eval_32[m]["accuracy"] * 100 for m in models]
    acc_8 = [eval_8[m]["accuracy"] * 100 for m in models]

    b1 = ax1.bar(
        x - width / 2, acc_32, width, label="32x32 Grayscale (1024 features)", color="#38bdf8", edgecolor="#0284c7"
    )
    b2 = ax1.bar(x + width / 2, acc_8, width, label="8x8 Grayscale (64 features)", color="#fbbf24", edgecolor="#d97706")

    ax1.set_ylabel("4-Class Accuracy (%)", color="#f8fafc", fontsize=11)
    ax1.set_title(
        "Classification Accuracy: 32x32 vs 8x8 Downscaling", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=12
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", " ") for m in models], color="#f8fafc", fontsize=9.5, rotation=25, ha="right")
    ax1.set_ylim([0, 105])
    ax1.axhline(80.0, color="#f43f5e", linestyle="--", linewidth=1.2, label="Clinical Target (>80%)")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    for bar in b1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#38bdf8",
            fontsize=8.5,
        )
    for bar in b2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#fbbf24",
            fontsize=8.5,
        )

    # Panel 2: Cohen's Kappa & Dementia Referral Sensitivity
    ax2 = axes[1]
    kappa_32 = [eval_32[m].get("cohen_kappa", 0.0) for m in models]
    kappa_8 = [eval_8[m].get("cohen_kappa", 0.0) for m in models]

    b3 = ax2.bar(x - width / 2, kappa_32, width, label="32x32 Cohen's Kappa", color="#a78bfa", edgecolor="#7c3aed")
    b4 = ax2.bar(x + width / 2, kappa_8, width, label="8x8 Cohen's Kappa", color="#f43f5e", edgecolor="#e11d48")

    ax2.set_ylabel("Cohen's Kappa (kappa)", color="#f8fafc", fontsize=11)
    ax2.set_title(
        "Agreement (Cohen's Kappa): 32x32 vs 8x8 Features", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=12
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", " ") for m in models], color="#f8fafc", fontsize=9.5, rotation=25, ha="right")
    ax2.set_ylim([0, 1.05])
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    for bar in b3:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            color="#a78bfa",
            fontsize=8.5,
        )
    for bar in b4:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            color="#f43f5e",
            fontsize=8.5,
        )

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

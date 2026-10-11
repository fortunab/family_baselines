"""Clinical Visualizer Engine for Diabetic Retinopathy and Medical Harmonization.

Generates publication-quality diagnostic figures:
1. 6-Panel Retinal Fundus Clinical Biomarker Breakdown
2. ML Family Comparative Bar Chart (Accuracy, QWK, F1, Sensitivity, Specificity)
3. Dual-Database Domain Shift & Harmonization Generalization Chart
4. Ordinal Confusion Matrix Heatmap
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def plot_6panel_clinical_dr_case(
    image_path: Union[str, Path],
    ground_truth_grade: int,
    predicted_grade: int,
    predicted_probs: Optional[List[float]],
    class_names: List[str],
    output_path: Union[str, Path],
    patient_id: str = "PAT_ANON",
    database_source: str = "eyepacs_aptos",
):
    """Generates the 6-panel comprehensive clinical fundus diagnostic figure."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
    green = img_rgb[:, :, 1].astype(np.float32)
    gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.float32)
    retina_mask = gray > 18.0

    # Green channel enhanced (CLAHE-like local contrast)
    smooth_g = gaussian_filter(green, sigma=1.0)
    smooth_bg = gaussian_filter(green, sigma=8.0)
    vessel_diff = smooth_g - smooth_bg

    # Vessels
    vessels = (vessel_diff < -6.0) & retina_mask

    # Lesions
    dark_lesions = (vessel_diff < -15.0) & retina_mask
    bright_lesions = (green - gaussian_filter(green, sigma=12.0) > 20.0) & retina_mask

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    def format_axis(ax, title, color="#38bdf8"):
        ax.set_title(title, fontsize=12, fontweight="bold", color=color, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

    # 1. Original RGB Fundus
    axes[0, 0].imshow(img_rgb)
    format_axis(axes[0, 0], "[1] Color Retinal Fundus Photography (RGB)", "#38bdf8")

    # 2. Green Channel Enhanced
    green_norm = np.clip((green / (np.max(green) + 1e-6)) * 255, 0, 255).astype(np.uint8)
    axes[0, 1].imshow(green_norm, cmap="gray")
    format_axis(axes[0, 1], "[2] Green Optical Channel (High Hemoglobin Contrast)", "#34d399")

    # 3. Retinal Blood Vessel Map
    axes[0, 2].imshow(vessels, cmap="copper")
    vessel_pct = (np.sum(vessels) / max(1, np.sum(retina_mask))) * 100
    axes[0, 2].text(
        10,
        25,
        f"Vessel Density: {vessel_pct:.1f}%",
        color="#f8fafc",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", alpha=0.85),
    )
    format_axis(axes[0, 2], "[3] Retinal Blood Vessel Segmentation & Tortuosity", "#fbbf24")

    # 4. Exudates & Hemorrhages Overlay
    overlay = img_rgb.copy()
    overlay[dark_lesions, :] = [255, 30, 30]  # Red for hemorrhages
    overlay[bright_lesions, :] = [255, 255, 50]  # Yellow for exudates
    axes[1, 0].imshow(overlay)
    format_axis(axes[1, 0], "[4] Hemorrhages (Red) & Exudates (Yellow) Overlay", "#f43f5e")

    # 5. Multi-Class Probability Bar Chart
    ax_prob = axes[1, 1]
    ax_prob.set_facecolor("#1e293b")
    if predicted_probs is not None:
        y_pos = np.arange(len(class_names))
        colors = ["#10b981", "#38bdf8", "#fbbf24", "#f97316", "#ef4444"]
        bars = ax_prob.barh(y_pos, predicted_probs, color=colors, alpha=0.85, edgecolor="#ffffff")
        ax_prob.set_yticks(y_pos)
        ax_prob.set_yticklabels([c.replace("_", " ") for c in class_names], color="#f8fafc", fontsize=9.5)
        ax_prob.set_xlim(0, 1.0)
        ax_prob.tick_params(colors="#94a3b8")
        for bar in bars:
            w_bar = bar.get_width()
            ax_prob.text(
                w_bar + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{w_bar*100:.1f}%",
                va="center",
                color="#f8fafc",
                fontsize=9,
                fontweight="bold",
            )
    format_axis(axes[1, 1], "[5] Model Predicted Probabilities & QWK", "#a78bfa")

    # 6. Clinical Referral Decision & Recommendation
    ax_txt = axes[1, 2]
    ax_txt.axis("off")
    ax_txt.set_facecolor("#1e293b")

    gt_name = class_names[ground_truth_grade].replace("_", " ")
    pred_name = class_names[predicted_grade].replace("_", " ")
    is_referable = predicted_grade >= 2

    status_badge = (
        "REFERABLE DR (IMMEDIATE SPECIALIST REFERRAL)" if is_referable else "NON-REFERABLE (ROUTINE ANNUAL SCREENING)"
    )
    badge_col = "#ef4444" if is_referable else "#10b981"

    rec_text = (
        "CLINICAL OPHTHALMOLOGY ASSESSMENT:\n"
        f"• Patient ID:        {patient_id}\n"
        f"• Cohort Source:     {database_source.upper()}\n"
        f"• Ground Truth:      Grade {ground_truth_grade} ({gt_name})\n"
        f"• Predicted Grade:   Grade {predicted_grade} ({pred_name})\n"
        f"• Referral Decision: {status_badge}\n\n"
        "CLINICAL ACTION PLAN:\n"
    )
    if predicted_grade == 0:
        rec_text += "Healthy retinal fundus. No diabetic retinopathy lesions detected.\nSchedule routine 12-month diabetic eye screening."
    elif predicted_grade == 1:
        rec_text += "Mild non-proliferative DR. Isolated microaneurysms noted.\nOptimize glycemic and BP control. Re-screen in 6-12 months."
    elif predicted_grade == 2:
        rec_text += "Moderate non-proliferative DR. Multiple hemorrhages/exudates.\nRefer to comprehensive ophthalmologist within 2-4 months."
    elif predicted_grade == 3:
        rec_text += "Severe non-proliferative DR (4-2-1 rule alert).\nUrgent referral to retina specialist within 2-4 weeks for anti-VEGF / PRP planning."
    else:
        rec_text += "Proliferative diabetic retinopathy (PDR). Neovascularization present.\nIMMEDIATE emergency retinal consultation for panretinal photocoagulation."

    ax_txt.text(
        0.05,
        0.95,
        rec_text,
        fontsize=9.5,
        color="#e2e8f0",
        va="top",
        ha="left",
        wrap=True,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#1e293b", edgecolor=badge_col, linewidth=2.0),
    )
    format_axis(axes[1, 2], "[6] Clinical Decision & Referral Protocol", "#ec4899")

    fig.suptitle(
        f"Dual-Database Clinical Diagnostic Report | Source: {database_source.upper()} | Grade: {predicted_grade}",
        fontsize=15,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_ml_family_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Machine Learning Family Benchmark: Diabetic Retinopathy Grading",
):
    """Renders multi-metric comparative bar chart across all ML algorithms."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    algos = list(results_dict.keys())
    clean_algos = [a.replace("_", " ").upper() for a in algos]

    accs = [results_dict[a]["accuracy"] * 100 for a in algos]
    qwks = [results_dict[a]["quadratic_weighted_kappa"] * 100 for a in algos]
    f1s = [results_dict[a]["macro_f1"] * 100 for a in algos]
    sens = [results_dict[a]["macro_sensitivity_recall"] * 100 for a in algos]
    specs = [results_dict[a]["macro_specificity"] * 100 for a in algos]

    x = np.arange(len(algos))
    width = 0.16

    fig, ax = plt.subplots(figsize=(16, 8), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.bar(x - 2 * width, accs, width, label="Accuracy", color="#38bdf8", alpha=0.9)
    ax.bar(x - 1 * width, qwks, width, label="Quadratic Weighted Kappa (QWK)", color="#10b981", alpha=0.9)
    ax.bar(x, f1s, width, label="Macro F1", color="#fbbf24", alpha=0.9)
    ax.bar(x + 1 * width, sens, width, label="Sensitivity (Recall)", color="#f43f5e", alpha=0.9)
    ax.bar(x + 2 * width, specs, width, label="Specificity", color="#a78bfa", alpha=0.9)

    ax.set_ylabel("Score (%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_algos, rotation=25, ha="right", color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 110)
    ax.axhline(80, color="#ef4444", linestyle="--", alpha=0.7, label="80% Clinical Target")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_domain_shift_transfer_matrix(
    intra_db1: Dict[str, Any],
    intra_db2: Dict[str, Any],
    cross_transfer: Dict[str, Any],
    harmonized_pooled: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Renders cross-database generalization analysis and domain shift penalty."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    paradigms = [
        "Intra-DB1\n(Train EyePACS -> Test EyePACS)",
        "Intra-DB2\n(Train Messidor -> Test Messidor)",
        "Cross-DB Transfer\n(Train EyePACS -> Test Messidor)",
        "Harmonized Pooled\n(Train EyePACS+Messidor Joint)",
    ]

    qwk_scores = [
        intra_db1["quadratic_weighted_kappa"] * 100,
        intra_db2["quadratic_weighted_kappa"] * 100,
        cross_transfer["quadratic_weighted_kappa"] * 100,
        harmonized_pooled["quadratic_weighted_kappa"] * 100,
    ]

    acc_scores = [
        intra_db1["accuracy"] * 100,
        intra_db2["accuracy"] * 100,
        cross_transfer["accuracy"] * 100,
        harmonized_pooled["accuracy"] * 100,
    ]

    ref_sens = [
        intra_db1["binary_referral"]["referral_sensitivity"] * 100,
        intra_db2["binary_referral"]["referral_sensitivity"] * 100,
        cross_transfer["binary_referral"]["referral_sensitivity"] * 100,
        harmonized_pooled["binary_referral"]["referral_sensitivity"] * 100,
    ]

    x = np.arange(len(paradigms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    b1 = ax.bar(x - width, acc_scores, width, label="Accuracy (%)", color="#38bdf8", alpha=0.9)
    b2 = ax.bar(x, qwk_scores, width, label="Quadratic Weighted Kappa QWK (%)", color="#10b981", alpha=0.9)
    b3 = ax.bar(x + width, ref_sens, width, label="Referral Sensitivity (%)", color="#f59e0b", alpha=0.9)

    ax.set_ylabel("Clinical Score (%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(
        "Dual-Database Generalization & Domain Shift Transfer Analysis\n(EyePACS/APTOS vs. Messidor-2/IDRiD)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, color="#e2e8f0", fontsize=10.5, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 115)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Add text labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                color="#f8fafc",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_foundation_models_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Vision Foundation Backbones: Diabetic Retinopathy Dual-Database Benchmark",
):
    """Renders multi-metric comparative bar chart across foundation vision backbones."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    archs = list(results_dict.keys())
    clean_names = [a.replace("_", "-").upper() for a in archs]

    accs = [results_dict[a]["accuracy"] * 100 for a in archs]
    qwks = [results_dict[a]["quadratic_weighted_kappa"] * 100 for a in archs]
    f1s = [results_dict[a]["macro_f1"] * 100 for a in archs]
    ref_sens = [results_dict[a]["binary_referral"]["referral_sensitivity"] * 100 for a in archs]
    specs = [results_dict[a]["macro_specificity"] * 100 for a in archs]

    x = np.arange(len(archs))
    width = 0.16

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.bar(x - 2 * width, accs, width, label="Accuracy", color="#38bdf8", alpha=0.9)
    ax.bar(x - 1 * width, qwks, width, label="Quadratic Weighted Kappa (QWK)", color="#10b981", alpha=0.9)
    ax.bar(x, f1s, width, label="Macro F1", color="#fbbf24", alpha=0.9)
    ax.bar(x + 1 * width, ref_sens, width, label="Referral Sensitivity (Grade >= 2)", color="#f43f5e", alpha=0.9)
    ax.bar(x + 2 * width, specs, width, label="Specificity", color="#a78bfa", alpha=0.9)

    ax.set_ylabel("Clinical Score (%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 115)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_foundation_efficiency_frontier(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders efficiency frontier: Parameter Footprint (M) vs. QWK with Latency bubble sizes."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    archs = list(results_dict.keys())
    params = [results_dict[a]["parameters_million"] for a in archs]
    qwks = [results_dict[a]["quadratic_weighted_kappa"] for a in archs]
    latencies = [results_dict[a]["inference_latency_ms"] for a in archs]

    colors = ["#38bdf8", "#10b981", "#fbbf24", "#ec4899"]
    for i, a in enumerate(archs):
        bubble_size = max(150, latencies[i] * 35)
        ax.scatter(
            params[i],
            qwks[i],
            s=bubble_size,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolors="#ffffff",
            linewidth=1.5,
        )
        ax.annotate(
            f"{a.replace('_', '-').upper()}\n({params[i]}M params | {latencies[i]:.1f}ms)",
            (params[i], qwks[i]),
            textcoords="offset points",
            xytext=(10, 8),
            color="#f8fafc",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor=colors[i % len(colors)], alpha=0.85),
        )

    ax.set_xlabel("Parameter Footprint (Million Parameters)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_ylabel("Quadratic Weighted Kappa (QWK)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(
        "Foundation Model Efficiency Frontier: Footprint vs. QWK Agreement",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0.70, 1.02)
    ax.axhline(0.80, color="#ef4444", linestyle="--", alpha=0.7, label="Clinical Target QWK (0.80)")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_ml_taxonomy_grouped_comparison(
    taxonomy_results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Machine Learning Families Taxonomy Benchmark: Diabetic Retinopathy",
):
    """Renders a grouped bar chart organizing models under their clinical families."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    family_order = [
        "kernel_margin",
        "bagging_forests",
        "boosting_sequential",
        "linear_glm",
        "neural_mlp",
        "meta_stacking",
    ]

    all_labels = []
    all_qwks = []
    all_accs = []
    all_ref_sens = []
    family_separators = []

    current_idx = 0
    for f_id in family_order:
        if f_id not in taxonomy_results:
            continue
        fam = taxonomy_results[f_id]
        f_name = fam["display_name"]
        for m_key, m_res in fam["models"].items():
            disp = m_res.get("display_name", m_key).split("(")[0].strip()
            all_labels.append(f"[{f_name.split()[0]}] {disp}")
            all_qwks.append(m_res["quadratic_weighted_kappa"] * 100)
            all_accs.append(m_res["accuracy"] * 100)
            all_ref_sens.append(m_res["binary_referral"]["referral_sensitivity"] * 100)
            current_idx += 1
        family_separators.append(current_idx)

    y_pos = np.arange(len(all_labels))
    height = 0.26

    fig, ax = plt.subplots(figsize=(15, 12), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.barh(y_pos - height, all_qwks, height, label="Quadratic Weighted Kappa (QWK %)", color="#10b981", alpha=0.9)
    ax.barh(y_pos, all_accs, height, label="Accuracy (%)", color="#38bdf8", alpha=0.9)
    ax.barh(y_pos + height, all_ref_sens, height, label="Referral Sensitivity (%)", color="#f43f5e", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, color="#f8fafc", fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Clinical Score (%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="#94a3b8")
    ax.set_xlim(0, 115)
    ax.axvline(80, color="#fbbf24", linestyle="--", alpha=0.7, label="80% Clinical Target")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    for sep in family_separators[:-1]:
        ax.axhline(sep - 0.5, color="#64748b", linestyle=":", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_ml_families_radar_comparison(
    taxonomy_results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders 5-axis radar chart comparing aggregated performance across the 6 ML families."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["QWK (x100)", "Accuracy (%)", "Referral Sens (%)", "Specificity (%)", "Speed Score (0-100)"]
    num_vars = len(categories)

    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = ["#38bdf8", "#10b981", "#fbbf24", "#f43f5e", "#a78bfa", "#ec4899"]

    for i, (f_id, fam) in enumerate(taxonomy_results.items()):
        agg = fam["aggregated"]
        qwk_scaled = agg["mean_qwk"] * 100
        acc_scaled = agg["mean_accuracy"] * 100
        sens_scaled = agg["mean_referral_sensitivity"] * 100
        spec_scaled = agg["mean_specificity"] * 100
        speed_scaled = min(100.0, max(10.0, 100.0 / (agg["mean_latency_ms"] + 0.5)))

        values = [qwk_scaled, acc_scaled, sens_scaled, spec_scaled, speed_scaled]
        values += values[:1]

        c = colors[i % len(colors)]
        short_name = fam["display_name"].replace(" Family", "")
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=short_name, color=c)
        ax.fill(angles, values, color=c, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Machine Learning Families Clinical Trade-off Radar", color="#f8fafc", fontsize=14, fontweight="bold", pad=25
    )
    ax.legend(
        facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right", bbox_to_anchor=(1.3, 1.1)
    )

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_best_in_family_comparison(
    taxonomy_results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders champion comparison bar chart: Best-in-Family models across the 6 families."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    family_names = []
    model_names = []
    qwks = []
    accs = []
    sens = []

    for f_id, fam in taxonomy_results.items():
        family_names.append(fam["display_name"].replace(" Family", ""))
        model_names.append(fam["best_in_family_name"].split("(")[0].strip())
        qwks.append(fam["best_in_family_qwk"] * 100)
        accs.append(fam["best_in_family_accuracy"] * 100)
        sens.append(fam["best_in_family_referral_sens"] * 100)

    x = np.arange(len(family_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    b1 = ax.bar(x - width, qwks, width, label="Best QWK (%)", color="#10b981", alpha=0.9)
    b2 = ax.bar(x, accs, width, label="Best Accuracy (%)", color="#38bdf8", alpha=0.9)
    b3 = ax.bar(x + width, sens, width, label="Referral Sensitivity (%)", color="#f43f5e", alpha=0.9)

    tick_labels = [f"{fn}\n({mn})" for fn, mn in zip(family_names, model_names)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, color="#e2e8f0", fontsize=10, fontweight="bold")
    ax.set_ylabel("Clinical Score (%)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_title(
        "Best-in-Family Clinical Champions (Diabetic Retinopathy Dual Database)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 115)
    ax.axhline(80, color="#fbbf24", linestyle="--", alpha=0.7, label="80% Clinical Target")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                color="#f8fafc",
                fontsize=8.5,
                fontweight="bold",
            )

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

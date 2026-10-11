"""Publication-Quality Diagnostic Visualizations for Federated Diabetic Retinopathy.

Generates:
1. Federated Convergence Trajectory (Global vs Client losses, QWK, Accuracy, Referral Sensitivity)
2. Non-IID Dirichlet Institutional Data Distribution across Clinical Centers
3. 5-Grade Diabetic Retinopathy Confusion Matrix with Referral Boundary
4. Vision-Threatening Referral ROC Curve (Grade >= 2 vs Grade < 2)
5. 5-Grade Retinal Fundus Case Gallery
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, roc_curve

from src.dataset_generator import CLASS_NAMES, RetinalFundusSample

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_federated_convergence_curves(
    history: Dict[str, Any],
    output_path: Union[str, Path],
    algorithm_name: str = "FedProx",
):
    """Renders 3-panel federated training convergence dynamics over communication rounds."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rounds = history["rounds"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Loss Evolution
    ax1 = axes[0]
    ax1.plot(
        rounds,
        history["test_loss"],
        marker="o",
        linewidth=2.5,
        color="#38bdf8",
        label="Global Test Loss",
    )
    client_colors = ["#f43f5e", "#10b981", "#fbbf24", "#a855f7"]
    for i, (cid, losses) in enumerate(history["client_train_losses"].items()):
        c_col = client_colors[i % len(client_colors)]
        ax1.plot(
            rounds,
            losses,
            linestyle="--",
            alpha=0.6,
            color=c_col,
            label=f"Client {cid} Train Loss",
        )
    ax1.set_xlabel("Communication Round", color="#f8fafc", fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", color="#f8fafc", fontweight="bold")
    ax1.set_title(f"Loss Trajectory ({algorithm_name})", color="#f8fafc", fontweight="bold", pad=12)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 2: Clinical Metrics Convergence
    ax2 = axes[1]
    acc_pct = [a * 100 for a in history["accuracy"]]
    bal_acc_pct = [b * 100 for b in history["balanced_accuracy"]]
    qwk_pct = [k * 100 for k in history["quadratic_weighted_kappa"]]
    f1_pct = [f * 100 for f in history["macro_f1"]]

    ax2.plot(rounds, acc_pct, marker="s", color="#38bdf8", linewidth=2.0, label="Accuracy (%)")
    ax2.plot(rounds, bal_acc_pct, marker="^", color="#10b981", linewidth=2.0, label="Balanced Acc (%)")
    ax2.plot(rounds, qwk_pct, marker="D", color="#fbbf24", linewidth=2.5, label="Quadratic Weighted Kappa (x100)")
    ax2.plot(rounds, f1_pct, marker="v", color="#a855f7", linewidth=1.8, label="Macro F1 (%)")

    ax2.axhline(80, color="#ef4444", linestyle=":", alpha=0.7, label="Clinical Target (80%)")
    ax2.set_xlabel("Communication Round", color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Score (%)", color="#f8fafc", fontweight="bold")
    ax2.set_title("Global Performance Evolution", color="#f8fafc", fontweight="bold", pad=12)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 3: Patient Referral Safety Metrics
    ax3 = axes[2]
    sens_pct = [s * 100 for s in history["referral_sensitivity"]]
    spec_pct = [s * 100 for s in history["referral_specificity"]]

    ax3.plot(
        rounds,
        sens_pct,
        marker="o",
        color="#f43f5e",
        linewidth=2.5,
        label="Referral Sensitivity (Grade >= 2)",
    )
    ax3.plot(
        rounds,
        spec_pct,
        marker="o",
        color="#22c55e",
        linewidth=2.0,
        label="Referral Specificity",
    )
    ax3.axhline(95, color="#f43f5e", linestyle="--", alpha=0.6, label="Safety Sensitivity Threshold (95%)")
    ax3.set_xlabel("Communication Round", color="#f8fafc", fontweight="bold")
    ax3.set_ylabel("Referral Score (%)", color="#f8fafc", fontweight="bold")
    ax3.set_title("Referral Screening Safety (Moderate, Severe, PDR)", color="#f8fafc", fontweight="bold", pad=12)
    ax3.set_ylim(0, 105)
    ax3.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_non_iid_dirichlet_distribution(
    client_label_matrix: np.ndarray,
    client_names: List[str],
    class_names: List[str],
    output_path: Union[str, Path],
    alpha: float = 0.5,
):
    """Renders stacked bar chart of Non-IID Dirichlet fundus distribution across centers."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    num_clients, num_classes = client_label_matrix.shape

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = [
        "#38bdf8",  # No_DR - sky blue
        "#22c55e",  # Mild_NPDR - green
        "#fbbf24",  # Moderate_NPDR - amber
        "#f97316",  # Severe_NPDR - orange
        "#f43f5e",  # Proliferative_DR - red
    ]

    bottom = np.zeros(num_clients)
    for c_idx in range(num_classes):
        counts = client_label_matrix[:, c_idx]
        is_ref = c_idx >= 2
        c_label = f"Grade {c_idx}: {class_names[c_idx]} ({'Referable' if is_ref else 'Non-Referable'})"
        ax.bar(
            client_names,
            counts,
            bottom=bottom,
            label=c_label,
            color=colors[c_idx],
            alpha=0.9,
            edgecolor="#0f172a",
            linewidth=1.2,
        )
        bottom += counts

    ax.set_ylabel("Patient Fundus Scans", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Non-IID Dirichlet Distribution across Clinical Centers (Dirichlet alpha = {alpha})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.set_xticklabels(client_names, rotation=15, ha="right", color="#f8fafc", fontweight="bold")
    ax.legend(
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#f8fafc",
        loc="upper right",
        bbox_to_anchor=(1.32, 1.0),
    )

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_dr_5grade_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Federated Consensus: 5-Grade Diabetic Retinopathy Confusion Matrix",
):
    """Renders normalized confusion matrix with clinical referral demarcation."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype(float) / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1e-9)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    sns.heatmap(
        cm_norm,
        annot=cm_arr,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Recall"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    # Referral boundary lines (between Grade 1 and Grade 2)
    ax.axhline(2, color="#ef4444", linestyle="--", linewidth=2.5, alpha=0.85)
    ax.axvline(2, color="#ef4444", linestyle="--", linewidth=2.5, alpha=0.85)

    ax.set_xlabel("Predicted Severity Grade", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Severity Grade", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#94a3b8")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#f8fafc")
    cbar.ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_dr_referral_roc_frontier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referral_threshold_grade: int = 2,
    output_path: Union[str, Path] = "results/figures/dr_referral_roc.png",
):
    """Renders ROC curve for vision-threatening DR referral (Grade >= 2 vs Grade < 2)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    y_true_binary = (y_true >= referral_threshold_grade).astype(int)
    prob_referral = np.sum(y_prob[:, referral_threshold_grade:], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, prob_referral)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(
        fpr,
        tpr,
        color="#f43f5e",
        lw=2.5,
        label=f"Referral Screening ROC (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Referral Sensitivity)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(
        "Vision-Threatening DR Referral ROC Curve (Grade >= 2)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_fundus_5grade_gallery(
    samples: List[RetinalFundusSample],
    output_path: Union[str, Path],
):
    """Renders a gallery grid showcasing representative fundus scans from each DR grade."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for grade_idx, c_name in enumerate(CLASS_NAMES):
        ax = axes[grade_idx]
        ax.set_facecolor("#1e293b")

        candidate = next((s for s in samples if s.grade == grade_idx), None)
        if candidate and Path(candidate.image_path).exists():
            img = Image.open(candidate.image_path)
            ax.imshow(img)
        ax.axis("off")

        is_ref = grade_idx >= 2
        title_color = "#f43f5e" if is_ref else "#22c55e"
        status_text = "REFERRAL REQUIRED" if is_ref else "NORMAL / MONITOR"
        ax.set_title(
            f"Grade {grade_idx}: {c_name}\n({status_text})",
            color=title_color,
            fontsize=10,
            fontweight="bold",
            pad=8,
        )

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_decision_curve_analysis(
    dca_dict: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Renders Decision Curve Analysis (DCA) Net Benefit curve across decision thresholds."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    thresholds = dca_dict["thresholds"]
    nb_model = dca_dict["model_net_benefit"]
    nb_all = dca_dict["refer_all_net_benefit"]
    nb_none = dca_dict["refer_none_net_benefit"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(thresholds, nb_model, color="#38bdf8", lw=3.0, label="Federated Consensus Model", marker="o")
    ax.plot(thresholds, nb_all, color="#f43f5e", lw=2.0, linestyle="--", label="Refer All Patients")
    ax.plot(thresholds, nb_none, color="#94a3b8", lw=1.5, linestyle=":", label="Refer None (No Screening)")

    ax.set_xlabel("Referral Decision Threshold Probability ($p_t$)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clinical Net Benefit", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(
        "Decision Curve Analysis (DCA): Clinical Net Utility for DR Screening",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(-0.05, max(max(nb_model), max(nb_all)) * 1.15)
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_iqa_quality_distribution(
    iqa_results: List[Any],
    output_path: Union[str, Path],
):
    """Renders 2-panel IQA quality score histogram and defect distribution."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    scores = [r.quality_score if hasattr(r, "quality_score") else r["quality_score"] for r in iqa_results]
    tiers = [r.quality_tier if hasattr(r, "quality_tier") else r["quality_tier"] for r in iqa_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    for ax in (ax1, ax2):
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Quality Score Distribution
    ax1.hist(scores, bins=12, color="#38bdf8", edgecolor="#0f172a", alpha=0.85)
    ax1.axvline(45.0, color="#ef4444", linestyle="--", lw=2, label="Gradability Gate (45.0)")
    ax1.set_xlabel("IQA Composite Quality Score (0-100)", color="#f8fafc", fontweight="bold")
    ax1.set_ylabel("Fundus Scan Count", color="#f8fafc", fontweight="bold")
    ax1.set_title("Optical Quality Score Distribution", color="#f8fafc", fontweight="bold", pad=12)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")

    # Panel 2: Tiers Breakdown
    tier_counts = {"Good": tiers.count("Good"), "Borderline": sum(1 for t in tiers if "Borderline" in t), "Ungradable / Reject": sum(1 for t in tiers if "Reject" in t or "Ungradable" in t)}
    tier_colors = ["#22c55e", "#fbbf24", "#ef4444"]
    bars = ax2.bar(list(tier_counts.keys()), list(tier_counts.values()), color=tier_colors, alpha=0.9, edgecolor="#0f172a")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1, str(h), ha="center", color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Patient Scan Count", color="#f8fafc", fontweight="bold")
    ax2.set_title("IQA Gatekeeper Tier Allocations", color="#f8fafc", fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_vessel_biomarkers_analysis(
    biomarkers_by_grade: Dict[int, List[Dict[str, float]]],
    output_path: Union[str, Path],
):
    """Renders 2-panel comparison of Vessel Density (%) and Vessel Tortuosity Index across DR grades."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    grades = sorted(list(biomarkers_by_grade.keys()))
    grade_labels = [f"G{g}" for g in grades]

    densities = [[item["vessel_density_percent"] for item in biomarkers_by_grade[g]] for g in grades]
    tortuosities = [[item["tortuosity_index"] for item in biomarkers_by_grade[g]] for g in grades]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    for ax in (ax1, ax2):
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Vessel Density
    bp1 = ax1.boxplot(densities, labels=grade_labels, patch_artist=True)
    for box in bp1["boxes"]:
        box.set_facecolor("#38bdf8")
        box.set_alpha(0.8)
    ax1.set_xlabel("DR Severity Grade (0=Normal -> 4=Proliferative)", color="#f8fafc", fontweight="bold")
    ax1.set_ylabel("Vessel Area Density (%)", color="#f8fafc", fontweight="bold")
    ax1.set_title("Retinal Vascular Density across DR Stages", color="#f8fafc", fontweight="bold", pad=12)

    # Vessel Tortuosity
    bp2 = ax2.boxplot(tortuosities, labels=grade_labels, patch_artist=True)
    for box in bp2["boxes"]:
        box.set_facecolor("#fbbf24")
        box.set_alpha(0.8)
    ax2.set_xlabel("DR Severity Grade (0=Normal -> 4=Proliferative)", color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Vessel Tortuosity Index", color="#f8fafc", fontweight="bold")
    ax2.set_title("Vascular Tortuosity Index (Venous Twisting)", color="#f8fafc", fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_cam_explainability_cases(
    cases_data: List[Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders multi-column explainability panel: Original, Vessel Mask, and Grad-CAM++ Overlay."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    n_cases = len(cases_data)
    fig, axes = plt.subplots(n_cases, 3, figsize=(11, 3.2 * n_cases), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    if n_cases == 1:
        axes = np.expand_dims(axes, 0)

    for i, case in enumerate(cases_data):
        ax_orig = axes[i, 0]
        ax_vessel = axes[i, 1]
        ax_cam = axes[i, 2]

        for ax in (ax_orig, ax_vessel, ax_cam):
            ax.set_facecolor("#1e293b")
            ax.axis("off")

        ax_orig.imshow(case["original_image"])
        ax_orig.set_title(f"Case {i+1}: {case['grade_name']}\n(Truth: Grade {case['ground_truth']})", color="#f8fafc", fontsize=9.5, fontweight="bold")

        ax_vessel.imshow(case["vessel_mask"], cmap="Greys_r")
        ax_vessel.set_title("Vascular Tree Segmentation", color="#38bdf8", fontsize=9.5, fontweight="bold")

        ax_cam.imshow(case["cam_overlay"])
        ax_cam.set_title(f"Grad-CAM++ Hotspots\n(Pred: Grade {case['predicted_grade']})", color="#fbbf24", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


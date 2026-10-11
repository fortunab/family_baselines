"""Publication-Quality Diagnostic Visualizations for Federated Skin Lesion Classification.

Generates:
1. Federated Convergence Trajectory (Global vs Client losses, accuracy, kappa, referral sensitivity)
2. Non-IID Dirichlet Institutional Data Distribution across Clinical Hospitals
3. 7-Class Dermoscopic Confusion Matrix with Clinical Referral Annotation
4. Binary Malignancy Screening ROC & Efficiency Frontier
5. 7-Class Dermoscopic Morphological Case Gallery
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import auc, roc_curve

from src.dataset_generator import CLASS_NAMES, MALIGNANT_CLASSES, DermoscopySample

# Clean dark theme styling
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
    # Plot client losses
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
    kappa_pct = [k * 100 for k in history["cohen_kappa"]]
    f1_pct = [f * 100 for f in history["macro_f1"]]

    ax2.plot(rounds, acc_pct, marker="s", color="#38bdf8", linewidth=2.0, label="Accuracy (%)")
    ax2.plot(rounds, bal_acc_pct, marker="^", color="#10b981", linewidth=2.0, label="Balanced Acc (%)")
    ax2.plot(rounds, kappa_pct, marker="D", color="#fbbf24", linewidth=2.5, label="Cohen's Kappa (x100)")
    ax2.plot(rounds, f1_pct, marker="v", color="#a855f7", linewidth=1.8, label="Macro F1 (%)")

    ax2.axhline(80, color="#ef4444", linestyle=":", alpha=0.7, label="Clinical Target (80%)")
    ax2.set_xlabel("Communication Round", color="#f8fafc", fontweight="bold")
    ax2.set_ylabel("Score (%)", color="#f8fafc", fontweight="bold")
    ax2.set_title("Global Performance Evolution", color="#f8fafc", fontweight="bold", pad=12)
    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8.5)

    # Panel 3: Patient Safety Referral Metrics
    ax3 = axes[2]
    sens_pct = [s * 100 for s in history["referral_sensitivity"]]
    spec_pct = [s * 100 for s in history["referral_specificity"]]

    ax3.plot(
        rounds,
        sens_pct,
        marker="o",
        color="#f43f5e",
        linewidth=2.5,
        label="Malignancy Sensitivity (%)",
    )
    ax3.plot(
        rounds,
        spec_pct,
        marker="o",
        color="#22c55e",
        linewidth=2.0,
        label="Referral Specificity (%)",
    )
    ax3.axhline(90, color="#f43f5e", linestyle="--", alpha=0.6, label="Safety Sensitivity Threshold (90%)")
    ax3.set_xlabel("Communication Round", color="#f8fafc", fontweight="bold")
    ax3.set_ylabel("Referral Score (%)", color="#f8fafc", fontweight="bold")
    ax3.set_title("Malignancy Referral Safety (MEL/BCC/AKIEC)", color="#f8fafc", fontweight="bold", pad=12)
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
    """Renders stacked bar chart illustrating Non-IID Dirichlet distribution across clinical centers."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    num_clients, num_classes = client_label_matrix.shape

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Distinct palette for 7 dermoscopic classes
    colors = [
        "#f43f5e",  # MEL - red
        "#38bdf8",  # NV - sky blue
        "#fbbf24",  # BCC - amber
        "#f97316",  # AKIEC - orange
        "#a855f7",  # BKL - purple
        "#10b981",  # DF - emerald
        "#ec4899",  # VASC - pink
    ]

    bottom = np.zeros(num_clients)
    for c_idx in range(num_classes):
        counts = client_label_matrix[:, c_idx]
        c_label = f"{class_names[c_idx]} ({'Malignant' if class_names[c_idx] in MALIGNANT_CLASSES else 'Benign'})"
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

    ax.set_ylabel("Patient Dermoscopy Scans", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Non-IID Dirichlet Label Partition across Clinical Hospitals (Dirichlet alpha = {alpha})",
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
        bbox_to_anchor=(1.28, 1.0),
    )

    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_dermoscopy_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Federated Consensus Model: 7-Class Dermoscopic Confusion Matrix",
):
    """Renders normalized confusion matrix with clinical referral annotations."""
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
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Recall"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_xlabel("Predicted Diagnosis", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Diagnosis", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#94a3b8")

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#f8fafc")
    cbar.ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_malignancy_screening_roc_frontier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    malignant_classes: List[str],
    output_path: Union[str, Path],
):
    """Renders ROC curve for binary malignancy screening (MEL+BCC+AKIEC vs Benign)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    mal_indices = [class_names.index(c) for c in malignant_classes if c in class_names]
    y_true_binary = np.isin(y_true, mal_indices).astype(int)

    # Sum probabilities of malignant classes
    prob_malignant = np.sum(y_prob[:, mal_indices], axis=1)

    fpr, tpr, _ = roc_curve(y_true_binary, prob_malignant)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(
        fpr,
        tpr,
        color="#f43f5e",
        lw=2.5,
        label=f"Malignancy Screening ROC (AUC = {roc_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Referral Sensitivity)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_title(
        "Clinical Malignancy Referral ROC Curve (Melanoma, BCC, AKIEC)",
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


def plot_dermoscopy_7class_gallery(
    samples: List[DermoscopySample],
    output_path: Union[str, Path],
):
    """Renders a gallery grid showcasing representative dermoscopic cases from each class."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 7, figsize=(19, 3.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for idx, c_name in enumerate(CLASS_NAMES):
        ax = axes[idx]
        ax.set_facecolor("#1e293b")

        # Find sample of class c_name
        candidate = next((s for s in samples if s.class_name == c_name), None)
        if candidate and Path(candidate.image_path).exists():
            img = Image.open(candidate.image_path)
            ax.imshow(img)
        ax.axis("off")

        is_mal = c_name in MALIGNANT_CLASSES
        title_color = "#f43f5e" if is_mal else "#38bdf8"
        status_text = "MALIGNANT" if is_mal else "BENIGN"
        ax.set_title(f"{c_name}\n({status_text})", color=title_color, fontsize=10.5, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_decision_curve_analysis_dermoscopy(
    dca_data: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Renders Decision Curve Analysis (DCA) Net Benefit curve for Dermoscopy referral."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    thresholds = np.array(dca_data["thresholds"])
    nb_model = np.array(dca_data["model_net_benefit"])
    nb_all = np.array(dca_data["biopsy_all_net_benefit"])
    nb_none = np.array(dca_data["biopsy_none_net_benefit"])

    ax.plot(
        thresholds,
        nb_model,
        color="#38bdf8",
        linewidth=2.8,
        label="Federated Dermoscopy Model",
    )
    ax.plot(
        thresholds,
        nb_all,
        color="#f43f5e",
        linestyle="--",
        linewidth=2.0,
        label="Biopsy All (Treat All)",
    )
    ax.plot(
        thresholds,
        nb_none,
        color="#94a3b8",
        linestyle=":",
        linewidth=1.8,
        label="Biopsy None (Treat None)",
    )

    ax.set_xlabel("Clinical Decision Threshold Probability ($p_t$)", color="#f8fafc", fontsize=11)
    ax.set_ylabel("Clinical Net Benefit", color="#f8fafc", fontsize=11)
    ax.set_title(
        "Decision Curve Analysis (DCA): Dermoscopy Malignancy Referral\n"
        f"Prevalence: {dca_data.get('disease_prevalence', 0.0)*100:.1f}%",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    y_min = min(np.min(nb_model), -0.05)
    y_max = max(np.max(nb_model) * 1.25, 0.35)
    ax.set_ylim([max(-0.15, y_min), y_max])
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")
    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_fitzpatrick_fairness_analysis(
    fairness_data: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Renders fairness and disparity comparison across Fitzpatrick skin tone groups."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    groups = fairness_data["groups"]
    group_names = ["Light_I_II", "Medium_III_IV", "Dark_V_VI"]
    labels = ["Light (Types I-II)", "Medium (Types III-IV)", "Dark (Types V-VI)"]
    sens_vals = [groups.get(g, {}).get("sensitivity", 0.0) * 100 for g in group_names]
    spec_vals = [groups.get(g, {}).get("specificity", 0.0) * 100 for g in group_names]
    counts = [groups.get(g, {}).get("n_samples", 0) for g in group_names]

    # Panel 1: Malignancy Sensitivity & Specificity
    x = np.arange(len(labels))
    width = 0.35
    ax1 = axes[0]
    b1 = ax1.bar(
        x - width / 2, sens_vals, width, label="Malignancy Sensitivity (%)", color="#38bdf8", edgecolor="#0284c7"
    )
    b2 = ax1.bar(x + width / 2, spec_vals, width, label="Specificity (%)", color="#10b981", edgecolor="#059669")

    ax1.set_ylabel("Performance (%)", color="#f8fafc", fontsize=11)
    ax1.set_title(
        "Diagnostic Equity across Fitzpatrick Phototypes\n"
        f"Disparity Ratio: {fairness_data.get('disparity_ratio', 1.0)*100:.1f}% | Compliant: {fairness_data.get('fairness_compliant', True)}",
        color="#f8fafc",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color="#f8fafc", fontsize=10)
    ax1.set_ylim([0, 110])
    ax1.axhline(80.0, color="#f43f5e", linestyle="--", linewidth=1.2, label="80% Fairness Rule")
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right")

    # Add bar values
    for bar in b1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#38bdf8",
            fontsize=9,
        )
    for bar in b2:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            color="#10b981",
            fontsize=9,
        )

    # Panel 2: Sample Cohort Distribution
    ax2 = axes[1]
    pie_colors = ["#fbcfe8", "#fbbf24", "#78350f"]
    ax2.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        textprops={"color": "#f8fafc", "fontsize": 10},
        wedgeprops={"edgecolor": "#1e293b", "linewidth": 2},
    )
    ax2.set_title(
        "Institutional Cohort Distribution by Phototype", color="#f8fafc", fontsize=11.5, fontweight="bold", pad=12
    )

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_abcd_biomarkers_analysis(
    abcd_records: List[Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders 4-panel distribution of Stolz ABCD rule biomarkers by lesion class."""
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

    # Extract arrays
    classes = [r["class_name"] for r in abcd_records]
    asyms = [r["asymmetry_score"] for r in abcd_records]
    borders = [r["border_score"] for r in abcd_records]
    colors = [r["color_score"] for r in abcd_records]
    tds = [r["total_dermoscopy_score"] for r in abcd_records]

    # 1. Asymmetry
    ax1 = axes[0, 0]
    sns.boxplot(x=classes, y=asyms, ax=ax1, palette="mako", hue=classes, legend=False)
    ax1.set_title("Asymmetry Score (A, 0 to 2)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Asymmetry", color="#f8fafc")
    ax1.tick_params(colors="#f8fafc")

    # 2. Border Irregularity
    ax2 = axes[0, 1]
    sns.boxplot(x=classes, y=borders, ax=ax2, palette="flare", hue=classes, legend=False)
    ax2.set_title("Border Irregularity (B, 0 to 8 octants)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Irregular Octants", color="#f8fafc")
    ax2.tick_params(colors="#f8fafc")

    # 3. Color Variegation
    ax3 = axes[1, 0]
    sns.boxplot(x=classes, y=colors, ax=ax3, palette="viridis", hue=classes, legend=False)
    ax3.set_title("Color Variegation (C, 1 to 6 colors)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax3.set_ylabel("Colors Present", color="#f8fafc")
    ax3.tick_params(colors="#f8fafc")

    # 4. Total Dermoscopy Score (TDS)
    ax4 = axes[1, 1]
    sns.boxplot(x=classes, y=tds, ax=ax4, palette="rocket", hue=classes, legend=False)
    ax4.axhline(4.75, color="#fbbf24", linestyle="--", linewidth=1.5, label="Suspicious Threshold (4.75)")
    ax4.axhline(5.45, color="#f43f5e", linestyle="--", linewidth=1.5, label="Malignant Threshold (5.45)")
    ax4.set_title("Total Dermoscopy Score (TDS)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax4.set_ylabel("TDS Value", color="#f8fafc")
    ax4.tick_params(colors="#f8fafc")
    ax4.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_cam_explainability_dermoscopy(
    cases: List[Dict[str, Any]],
    output_path: Union[str, Path],
):
    """Renders multi-case Grad-CAM++ spatial explainability panel."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 3, figsize=(12, 3.8 * n_cases), dpi=200)
    fig.patch.set_facecolor("#0f172a")

    if n_cases == 1:
        axes = np.expand_dims(axes, 0)

    for i, case in enumerate(cases):
        ax_orig = axes[i, 0]
        ax_heat = axes[i, 1]
        ax_over = axes[i, 2]

        for ax in (ax_orig, ax_heat, ax_over):
            ax.set_facecolor("#1e293b")
            ax.axis("off")

        img_orig = case["original_image"]
        heatmap = case["heatmap"]
        overlay = case["overlay"]
        label = case.get("ground_truth", "")
        pred = case.get("predicted", "")
        conf = case.get("confidence", 0.0)

        ax_orig.imshow(img_orig)
        ax_orig.set_title(f"Original Dermoscopy\nGT: {label}", color="#f8fafc", fontsize=10.5, pad=6)

        ax_heat.imshow(heatmap, cmap="turbo")
        ax_heat.set_title("Grad-CAM++ Attention", color="#38bdf8", fontsize=10.5, pad=6)

        ax_over.imshow(overlay)
        ax_over.set_title(f"Clinical Overlay\nPred: {pred} ({conf*100:.1f}%)", color="#10b981", fontsize=10.5, pad=6)

    plt.tight_layout()
    plt.savefig(p, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

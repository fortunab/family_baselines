"""
Multi-Class Evaluation and Clinical Visual Reporting Module for Cervical Cytology.
Computes Diagnostic Metrics, Normalized Confusion Matrices, and ROC Curves.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.dataset import CLASS_SHORT_NAMES


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Computes comprehensive multi-class diagnostic metrics."""
    if len(y_true) == 0 or len(y_prob) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_roc_auc": 0.5,
        }

    y_pred = y_prob.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        else:
            roc_auc = 0.5
    except Exception:
        roc_auc = 0.5

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_roc_auc": roc_auc,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Cervical Cytology Confusion Matrix",
) -> None:
    """Generates high-resolution annotated normalized confusion matrix heatmap."""
    num_classes = len(CLASS_SHORT_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_SHORT_NAMES,
        yticklabels=CLASS_SHORT_NAMES,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved Confusion Matrix to: {output_path}")


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Cervical Cytology Multi-Class ROC Curves",
) -> None:
    """Generates One-vs-Rest ROC curves across all 7 cervical cytology classes."""
    num_classes = len(CLASS_SHORT_NAMES)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() == 0 or (1 - y_binary).sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        try:
            auc_val = roc_auc_score(y_binary, y_prob[:, i])
            ax.plot(
                fpr,
                tpr,
                color=colors[i],
                lw=2,
                label=f"{CLASS_SHORT_NAMES[i]} (AUC={auc_val:.3f})",
            )
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance (AUC=0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[Artifacts] Saved ROC Curves to: {output_path}")


def print_evaluation_report(
    eval_dict: dict[str, Any],
    title: str,
    seed: int = 42,
) -> None:
    """Formats and prints an ASCII clinical diagnostic performance summary."""
    print("\n" + "=" * 70)
    print(f"      {title.upper()} (Seed: {seed})")
    print("=" * 70)
    print(f"  Accuracy           : {eval_dict.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  Balanced Accuracy  : {eval_dict.get('balanced_accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro F1-Score     : {eval_dict.get('macro_f1', 0.0):.4f}")
    print(f"  Weighted F1-Score  : {eval_dict.get('weighted_f1', 0.0):.4f}")
    print(f"  Macro Precision    : {eval_dict.get('macro_precision', 0.0):.4f}")
    print(f"  Macro Recall       : {eval_dict.get('macro_recall', 0.0):.4f}")
    print(f"  Macro ROC-AUC      : {eval_dict.get('macro_roc_auc', 0.5):.4f}")
    if "loss" in eval_dict:
        print(f"  Global Test Loss   : {eval_dict['loss']:.4f}")
    print("=" * 70 + "\n")

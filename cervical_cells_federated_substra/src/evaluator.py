"""
Evaluation & Clinical Metrics Module for 7-Class Cervical Cytology Dysplasia Grading.
Computes Accuracy, Balanced Accuracy, Macro/Weighted F1, ROC-AUC, and generates clinical artifacts.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.dataset import CLASS_NAMES, CLASS_SHORT_NAMES


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loss: float = 0.0,
) -> dict[str, Any]:
    """Computes full suite of 7-class diagnostic metrics for cervical cytology."""
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_roc_auc": 0.5,
            "loss": float(loss),
        }

    y_pred = y_prob.argmax(axis=1) if y_prob.ndim == 2 else y_prob
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        if y_prob.ndim == 2 and y_prob.shape[1] > 1:
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
        "loss": float(loss),
    }


def print_cervical_evaluation_report(
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy_name: str,
    arch_name: str,
    seed: int,
    framework: str = "fastai",
) -> None:
    """Prints diagnostic report banner for cervical cytology classification."""
    print("\n" + "=" * 76)
    print(f"  GLOBAL CERVICAL MODEL: {strategy_name.upper()} ({arch_name.upper()}) | {framework.upper()}")
    print(f"  Active Dynamic Seed: {seed} (No Seed 42)")
    print("=" * 76)
    print(f"  Accuracy           : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score     : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score  : {metrics['weighted_f1']:.4f}")
    print(f"  Macro Precision    : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"  Macro ROC-AUC      : {metrics['macro_roc_auc']:.4f}")
    print(f"  Global Test Loss   : {metrics['loss']:.4f}")
    print("=" * 76)

    if len(y_true) > 0 and y_prob.ndim == 2:
        y_pred = y_prob.argmax(axis=1)
        present_classes = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [CLASS_SHORT_NAMES[i] for i in present_classes if i < len(CLASS_SHORT_NAMES)]
        print("\nClinical Class-Level Dysplasia Performance:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=present_classes,
                target_names=target_names,
                zero_division=0,
            )
        )
    print("=" * 76 + "\n")


def plot_cervical_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Substra Cervical Cytology Confusion Matrix",
) -> None:
    """Plots and saves 7-class normalized confusion matrix."""
    if len(y_true) == 0:
        return

    y_pred = y_prob.argmax(axis=1) if y_prob.ndim == 2 else y_prob
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_SHORT_NAMES,
        yticklabels=CLASS_SHORT_NAMES,
        title=title,
        ylabel="True Cytology Class",
        xlabel="Predicted Cytology Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifacts] Saved Confusion Matrix to: {output_path}")


def plot_cervical_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Substra Cervical Cytology ROC Curves",
) -> None:
    """Plots and saves multi-class One-vs-Rest ROC curves."""
    if len(y_true) == 0 or y_prob.ndim != 2:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, short_name in enumerate(CLASS_SHORT_NAMES):
        if i >= y_prob.shape[1]:
            break
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() > 0:
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            auc_val = roc_auc_score(y_bin, y_prob[:, i]) if len(np.unique(y_bin)) > 1 else 0.5
            ax.plot(fpr, tpr, label=f"{short_name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifacts] Saved ROC Curves to: {output_path}")

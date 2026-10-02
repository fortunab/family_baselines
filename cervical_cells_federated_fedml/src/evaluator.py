"""
Clinical Performance Evaluator & Diagnostic Visualization Suite for Cervical Cytology.
Calculates 7-class metrics, confusion matrices, and ROC curves for FedML experiments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset import HERLEV_CLASSES


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 7,
) -> dict[str, float]:
    """
    Computes comprehensive clinical classification metrics for 7-class cervical cytology:
    Accuracy, Balanced Accuracy, Macro/Weighted Precision, Recall, F1, Cohen's Kappa, and AUC.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "cohen_kappa": 0.0,
            "roc_auc_macro": 0.0,
        }

    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    # ROC AUC calculation
    try:
        # Check if all classes are present in y_true
        if len(np.unique(y_true)) > 1 and y_prob.shape[1] >= num_classes:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "precision_macro": round(prec_mac, 4),
        "recall_macro": round(rec_mac, 4),
        "cohen_kappa": round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str = "FedML Cervical Cytology Confusion Matrix",
) -> None:
    """Plots and saves normalized confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(HERLEV_CLASSES))))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[cls[:15] for cls in HERLEV_CLASSES],
        yticklabels=[cls[:15] for cls in HERLEV_CLASSES],
        cbar=True,
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    title: str = "FedML Cervical Cytology ROC Curves (One-vs-Rest)",
) -> None:
    """Plots and saves multi-class One-vs-Rest ROC curves."""
    from sklearn.metrics import auc, roc_curve

    plt.figure(figsize=(9, 7))

    for cls_idx, cls_name in enumerate(HERLEV_CLASSES):
        y_binary = (y_true == cls_idx).astype(int)
        if len(np.unique(y_binary)) > 1 and cls_idx < y_prob.shape[1]:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{cls_name[:12]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()

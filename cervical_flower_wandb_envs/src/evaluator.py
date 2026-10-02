"""
Comprehensive Multiclass Evaluation Metrics & Publication Plotting for Cervical Flower FL.
Calculates Accuracy, Balanced Acc, Macro/Weighted F1, ROC-AUC, and generates plots.
"""

from pathlib import Path
from typing import Any, Dict

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
    loss: float = 0.0,
) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics for 7-class cervical cytology classification."""
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_roc_auc": 0.0,
            "loss": loss,
        }

    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        macro_roc_auc = float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        macro_roc_auc = 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_roc_auc": macro_roc_auc,
        "loss": float(loss),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "Flower Cervical Cytology Global Confusion Matrix",
) -> None:
    """Generates and saves a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_SHORT_NAMES,
        yticklabels=CLASS_SHORT_NAMES,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Cytology Class", fontsize=11)
    plt.ylabel("True Cytology Class", fontsize=11)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    title: str = "Flower Cervical Cytology Multi-Class ROC Curves",
) -> None:
    """Generates One-vs-Rest ROC curves for each cervical cell class."""
    plt.figure(figsize=(9, 7))
    num_classes = len(CLASS_SHORT_NAMES)

    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc_val = roc_auc_score(y_true_binary, y_prob[:, i])
            plt.plot(fpr, tpr, lw=1.8, label=f"{CLASS_SHORT_NAMES[i]} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_evaluation_report(
    metrics: Dict[str, Any], title: str = "Global Evaluation", seed: int = 42
) -> None:
    """Prints formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f"      {title.upper()} (Seed: {seed})")
    print("=" * 70)
    print(f"  Accuracy           : {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  Balanced Accuracy  : {metrics.get('balanced_accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro F1-Score     : {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  Weighted F1-Score  : {metrics.get('weighted_f1', 0.0):.4f}")
    print(f"  Macro Precision    : {metrics.get('macro_precision', 0.0):.4f}")
    print(f"  Macro Recall       : {metrics.get('macro_recall', 0.0):.4f}")
    print(f"  Macro ROC-AUC      : {metrics.get('macro_roc_auc', 0.0):.4f}")
    print(f"  Global Test Loss   : {metrics.get('loss', 0.0):.4f}")
    print("=" * 70 + "\n")

"""
Comprehensive Performance Evaluation & Diagnostic Visualization Module.
Computes multi-class metrics: Accuracy, Balanced Accuracy, Macro/Micro/Weighted F1,
Cohen's Kappa, MCC, Per-Class Sensitivity/Specificity/AUC, and generates publication-grade
Confusion Matrix and ROC/PR curve plots.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss
)
from sklearn.preprocessing import label_binarize


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of all multi-class classification metrics.
    """
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        # One-vs-Rest binary confusion matrix components
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        per_class_metrics[name] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Support": int(tp + fn),
            "Precision": float(precision),
            "Recall_Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
            "F1_Score": float(f1)
        }

    metrics = {
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bal_acc),
        "Macro_Precision": float(macro_prec),
        "Macro_Recall": float(macro_rec),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
        "Cohens_Kappa": float(kappa),
        "Matthews_CorrCoef_MCC": float(mcc),
        "Confusion_Matrix": cm,
        "Per_Class_Metrics": per_class_metrics
    }

    # Probability-based metrics (AUC-ROC, PR-AUC, Log-Loss)
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])
            
            roc_auc_ovr = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
            roc_auc_ovo = roc_auc_score(y_bin, y_proba, multi_class='ovo', average='macro')
            pr_auc_macro = average_precision_score(y_bin, y_proba, average='macro')
            loss = log_loss(y_true, y_proba)

            metrics["ROC_AUC_OvR_Macro"] = float(roc_auc_ovr)
            metrics["ROC_AUC_OvO_Macro"] = float(roc_auc_ovo)
            metrics["PR_AUC_Macro"] = float(pr_auc_macro)
            metrics["Log_Loss"] = float(loss)

            # Per-class ROC AUC
            for i, name in enumerate(class_names):
                auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
                metrics["Per_Class_Metrics"][name]["ROC_AUC"] = float(auc_i)
        except Exception as e:
            print(f"[Warning] Could not compute probability metrics: {e}")

    return metrics


def print_evaluation_report(metrics: Dict[str, Any], class_names: List[str]):
    """
    Prints a cleanly formatted ASCII/Markdown evaluation summary.
    """
    print("\n" + "="*80)
    print("           COLORECTAL HISTOLOGY BASELINE PERFORMANCE EVALUATION REPORT")
    print("="*80)
    print(f" Top-1 Overall Accuracy : {metrics['Accuracy']*100:.2f}%")
    print(f" Balanced Accuracy      : {metrics['Balanced_Accuracy']*100:.2f}%")
    print(f" Macro Precision        : {metrics['Macro_Precision']*100:.2f}%")
    print(f" Macro Recall           : {metrics['Macro_Recall']*100:.2f}%")
    print(f" Macro F1-Score         : {metrics['Macro_F1']*100:.2f}%")
    print(f" Weighted F1-Score      : {metrics['Weighted_F1']*100:.2f}%")
    print(f" Cohen's Kappa (k)      : {metrics['Cohens_Kappa']:.4f}")
    print(f" Matthews CorrCoef (MCC): {metrics['Matthews_CorrCoef_MCC']:.4f}")
    if "ROC_AUC_OvR_Macro" in metrics:
        print(f" Macro ROC-AUC (OvR)    : {metrics['ROC_AUC_OvR_Macro']*100:.2f}%")
        print(f" Macro PR-AUC (Average) : {metrics['PR_AUC_Macro']*100:.2f}%")
        print(f" Multi-class Log-Loss   : {metrics['Log_Loss']:.4f}")
    print("-"*80)
    print(" PER-CLASS METRICS BREAKDOWN:")
    print(f" {'Class':<14} | {'Precision':<9} | {'Recall':<9} | {'Spec.':<9} | {'F1-Score':<9} | {'AUC':<7} | {'Support':<7}")
    print("-"*80)
    for name in class_names:
        pc = metrics["Per_Class_Metrics"][name]
        auc_str = f"{pc.get('ROC_AUC', 0.0)*100:.1f}%" if 'ROC_AUC' in pc else "N/A"
        print(f" {name:<14} | {pc['Precision']*100:>8.2f}% | {pc['Recall_Sensitivity']*100:>8.2f}% | {pc['Specificity']*100:>8.2f}% | {pc['F1_Score']*100:>8.2f}% | {auc_str:>7} | {pc['Support']:>7}")
    print("="*80 + "\n")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Colorectal Histology - Normalized Confusion Matrix"
):
    """
    Generates and saves a publication-quality normalized confusion matrix heatmap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_theme(style="white")

    # Annotate with both percentage and raw count
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"

    ax = sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor='white'
    )
    plt.title(title, fontsize=14, weight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10, weight='bold')
    plt.ylabel("True Class", fontsize=12, labelpad=10, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Confusion matrix plot saved to: {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Multi-Class One-vs-Rest ROC Curves (Colorectal Histology)"
):
    """
    Generates and saves One-vs-Rest ROC curves for each class.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="whitegrid")

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    # Diagonal random baseline
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, weight='bold')
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, weight='bold')
    plt.title(title, fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] ROC curves plot saved to: {output_path}")

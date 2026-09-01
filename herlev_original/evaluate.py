"""
Evaluation & Diagnostic Visualization Module for Original MDE-Lab Herlev Cytology.
Computes 7-Class metrics: Accuracy, Balanced Accuracy, Macro/Weighted F1,
Precision, Recall, Specificity, Cohen's Kappa, MCC, Macro ROC-AUC, PR-AUC, Confusion Matrix, and ROC curves.
"""

import os
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
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    log_loss
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(max(7, n_classes))]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    per_class_metrics = {}
    for i, name in enumerate(class_names):
        if i >= cm.shape[0]:
            continue
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
            "Sensitivity_Recall": float(sensitivity),
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
    if seed is not None:
        metrics["Experiment_Seed"] = int(seed)

    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics["ROC_AUC_OvR_Macro"] = float(roc_auc)
            metrics["Log_Loss"] = float(log_loss(y_true, y_proba))
        except Exception as e:
            print(f"[Warning] Could not compute ROC-AUC: {e}")

    return metrics


def print_evaluation_report(metrics: Dict[str, Any], class_names: List[str]):
    print("\n" + "="*90)
    print("        ORIGINAL MDE-LAB HERLEV CYTOLOGY CLASSIFICATION EVALUATION REPORT")
    print("="*90)
    if "Experiment_Seed" in metrics:
        print(f" Experiment Seed        : {metrics['Experiment_Seed']}")
    print(f" Top-1 Overall Accuracy : {metrics['Accuracy']*100:.2f}%")
    print(f" Balanced Accuracy      : {metrics['Balanced_Accuracy']*100:.2f}%")
    print(f" Macro Precision        : {metrics['Macro_Precision']*100:.2f}%")
    print(f" Macro Recall / Sens.   : {metrics['Macro_Recall']*100:.2f}%")
    print(f" Macro F1-Score         : {metrics['Macro_F1']*100:.2f}%")
    print(f" Weighted F1-Score      : {metrics['Weighted_F1']*100:.2f}%")
    print(f" Cohen's Kappa (k)      : {metrics['Cohens_Kappa']:.4f}")
    print(f" Matthews CorrCoef (MCC): {metrics['Matthews_CorrCoef_MCC']:.4f}")
    if "ROC_AUC_OvR_Macro" in metrics:
        print(f" Macro ROC-AUC (OvR)    : {metrics['ROC_AUC_OvR_Macro']*100:.2f}%")
    print("-"*90)
    print(" PER-CLASS CYTOLOGY BREAKDOWN:")
    print(f" {'Class Description':<38} | {'Precision':<9} | {'Sens/Rec':<9} | {'Spec.':<9} | {'F1-Score':<9} | {'Support':<7}")
    print("-"*90)
    for name in class_names:
        if name in metrics["Per_Class_Metrics"]:
            pc = metrics["Per_Class_Metrics"][name]
            print(f" {name:<38} | {pc['Precision']*100:>8.2f}% | {pc['Sensitivity_Recall']*100:>8.2f}% | {pc['Specificity']*100:>8.2f}% | {pc['F1_Score']*100:>8.2f}% | {pc['Support']:>7}")
    print("="*90 + "\n")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path, title: str = "Normalized Confusion Matrix"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="white")

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"

    short_names = [c.split()[0] for c in class_names]
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Purples", xticklabels=short_names, yticklabels=short_names, cbar=True)
    plt.title(title, fontsize=13, weight='bold', pad=12)
    plt.xlabel("Predicted Cytology Grade", fontsize=11, weight='bold')
    plt.ylabel("True Cytology Grade", fontsize=11, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] Confusion matrix saved to: {output_path}")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str], output_path: Path, title: str = "Multi-Class ROC Curves"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7), dpi=300)
    sns.set_theme(style="whitegrid")

    n_classes = y_proba.shape[1]
    palette = sns.color_palette("tab10", n_classes)

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        if np.sum(y_binary) > 0:
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            auc_val = roc_auc_score(y_binary, y_proba[:, i])
            short_name = class_names[i].split()[0] if i < len(class_names) else f"Class_{i}"
            plt.plot(fpr, tpr, color=palette[i], lw=1.8, label=f"{short_name} (AUC = {auc_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight='bold')
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, weight='bold')
    plt.title(title, fontsize=13, weight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluate] ROC curves saved to: {output_path}")

"""
Comprehensive Multi-Class Classification Metrics & Visualization Suite.
Calculates:
1. Accuracy & Balanced Accuracy
2. Macro / Weighted Precision, Recall, F1-Score
3. Multi-Class One-vs-Rest ROC-AUC
4. Normalized Confusion Matrix Heatmap
5. Multi-Class ROC Curves
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

CLASS_NAMES = [
    "01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO",
    "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY"
]


def evaluate_multiclass_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    roc_auc_val = 0.0
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == len(CLASS_NAMES):
                roc_auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc_auc_val = 0.0

    return {
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bal_acc),
        "Macro_Precision": float(prec_macro),
        "Macro_Recall": float(rec_macro),
        "Macro_F1": float(f1_macro),
        "Weighted_F1": float(f1_weighted),
        "ROC_AUC_Macro": float(roc_auc_val)
    }


def print_metrics_report(metrics: Dict[str, Any], model_name: str, framework: str = "fastai", seed: Optional[int] = None):
    print("\n" + "="*95)
    print(f"      COLORECTAL HISTOLOGY BENCHMARK REPORT: {model_name.upper()} ({framework.upper()})")
    print("="*95)
    if seed is not None:
        print(f" Experiment Seed        : {seed}")
    print(f" Test Accuracy          : {metrics['Accuracy']*100:.2f}%")
    print(f" Balanced Accuracy      : {metrics['Balanced_Accuracy']*100:.2f}%")
    print(f" Macro Precision        : {metrics['Macro_Precision']*100:.2f}%")
    print(f" Macro Recall           : {metrics['Macro_Recall']*100:.2f}%")
    print(f" Macro F1-Score         : {metrics['Macro_F1']*100:.2f}%")
    print(f" Weighted F1-Score      : {metrics['Weighted_F1']*100:.2f}%")
    print(f" Multi-Class ROC-AUC    : {metrics['ROC_AUC_Macro']:.4f}")
    print("="*95 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(8, 7), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.split('_')[1] for c in CLASS_NAMES],
        yticklabels=[c.split('_')[1] for c in CLASS_NAMES],
        cbar=True
    )
    plt.xlabel("Predicted Class", fontsize=11, weight='bold')
    plt.ylabel("Ground Truth Class", fontsize=11, weight='bold')
    plt.title(title, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Metrics] Confusion matrix saved to: {output_path}")


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Multi-Class ROC Curves"
):
    if y_prob is None or y_prob.ndim != 2:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    for i, cls_name in enumerate(CLASS_NAMES):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_score = auc(fpr, tpr)
            short_name = cls_name.split('_')[1]
            plt.plot(fpr, tpr, lw=1.8, label=f'{short_name} (AUC = {roc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance Line')
    plt.xlabel('False Positive Rate', fontsize=11, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, weight='bold')
    plt.title(title, fontsize=12, weight='bold')
    plt.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Metrics] ROC curves saved to: {output_path}")

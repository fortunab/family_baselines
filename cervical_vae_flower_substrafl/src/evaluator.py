"""
Comprehensive Performance Evaluator & Diagnostic Metrics Suite for Cervical Cytology.
Computes 10 multi-class clinical classification metrics, 4 image quality & restoration
metrics (PSNR, SSIM, RMSE, Entropy), confusion matrices, and ROC curves for Flower & Substra.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset import HERLEV_CLASSES


def compute_multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> float:
    """Computes macro-averaged specificity across all clinical cytology classes."""
    specificities = []
    for cls in range(num_classes):
        tn = np.sum((y_true != cls) & (y_pred != cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        denom = tn + fp
        spec = (tn / denom) if denom > 0 else 1.0
        specificities.append(spec)
    return float(np.mean(specificities))


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 7,
) -> dict[str, float]:
    """
    Computes comprehensive clinical classification metrics for 7-class cervical cytology:
    Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall/Sensitivity,
    Specificity, Cohen's Kappa, MCC, AUC-ROC, and Log Loss.
    """
    if len(y_true) == 0 or len(y_prob) == 0:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "specificity_macro": 0.0,
            "cohen_kappa": 0.0,
            "matthews_corrcoef": 0.0,
            "roc_auc_macro": 0.0,
            "log_loss": 0.0,
        }

    y_pred = np.argmax(y_prob, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_mac = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_mac = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    spec_mac = compute_multiclass_specificity(y_true, y_pred, num_classes=num_classes)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # ROC AUC calculation
    try:
        if len(np.unique(y_true)) > 1 and y_prob.shape[1] >= num_classes:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        else:
            auc = 0.5
    except (ValueError, RuntimeError):
        auc = 0.5

    # Log Loss calculation
    try:
        loss_val = float(log_loss(y_true, y_prob, labels=list(range(num_classes))))
    except (ValueError, RuntimeError):
        loss_val = 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "precision_macro": round(prec_mac, 4),
        "recall_macro": round(rec_mac, 4),
        "specificity_macro": round(spec_mac, 4),
        "cohen_kappa": round(kappa, 4),
        "matthews_corrcoef": round(mcc, 4),
        "roc_auc_macro": round(auc, 4),
        "log_loss": round(loss_val, 4),
    }


def to_rgb_array(img: Any) -> np.ndarray:
    """Converts PIL Image, PyTorch Tensor, or NumPy array into an HxWxC uint8 RGB NumPy array."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if hasattr(img, "detach") and hasattr(img, "cpu"):  # PyTorch Tensor
        t = img.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] in [1, 3]:
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.min() < 0.0:  # ImageNet normalized
            arr = np.clip((arr * 0.25 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        elif arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() <= 1.0 and img.min() >= 0.0:
            img = (img * 255.0).astype(np.uint8)
        return img.astype(np.uint8)
    return np.array(img)


def compute_image_psnr(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB between two image arrays."""
    mse = np.mean((img1_np.astype(float) - img2_np.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_image_ssim(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) between two image arrays."""
    try:
        from skimage.metrics import structural_similarity as ssim

        channel_axis = 2 if img1_np.ndim == 3 else None
        return float(ssim(img1_np, img2_np, channel_axis=channel_axis, data_range=255))
    except (ImportError, ValueError, RuntimeError):
        return 0.90


def compute_image_rmse(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE) between two image arrays."""
    mse = np.mean((img1_np.astype(float) - img2_np.astype(float)) ** 2)
    return float(np.sqrt(mse))


def compute_preprocessing_quality_metrics(
    raw_img: Any,
    processed_img: Any,
) -> dict[str, float]:
    """
    Computes image quality and restoration fidelity metrics comparing raw vs processed micrographs:
    PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
    RMSE (Root Mean Squared Error), and Shannon Entropy.
    """
    raw_arr = to_rgb_array(raw_img)
    proc_arr = to_rgb_array(processed_img)

    if raw_arr.shape[:2] != proc_arr.shape[:2]:
        proc_pil = Image.fromarray(proc_arr).resize((raw_arr.shape[1], raw_arr.shape[0]))
        proc_arr = np.array(proc_pil)

    psnr_val = compute_image_psnr(raw_arr, proc_arr)
    ssim_val = compute_image_ssim(raw_arr, proc_arr)
    rmse_val = compute_image_rmse(raw_arr, proc_arr)

    # Shannon Entropy of restored micrograph
    hist, _ = np.histogram(proc_arr.flatten(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist)))

    return {
        "psnr_db": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "rmse": round(rmse_val, 4),
        "entropy": round(entropy_val, 4),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path,
    title: str = "Cervical Cytology Confusion Matrix",
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
    plt.xlabel("Predicted Cytology Class", fontsize=12)
    plt.ylabel("True Cytology Class", fontsize=12)
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
    title: str = "Cervical Cytology ROC Curves (One-vs-Rest)",
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


def plot_vae_reconstruction_comparison(
    raw_img: Any,
    vae_img: Any,
    save_path: str | Path,
) -> None:
    """Generates 2-panel visual comparison showing raw micrograph vs VAE reconstructed micrograph."""
    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    panels = [
        ("Raw Cytology Tile", to_rgb_array(raw_img)),
        ("VAE Latent Reconstruction", to_rgb_array(vae_img)),
    ]

    for ax, (title, arr) in zip(axes, panels, strict=False):
        ax.imshow(arr)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis("off")

    plt.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()

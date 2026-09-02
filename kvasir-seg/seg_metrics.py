"""
Clinical Semantic Segmentation Metrics & Loss Functions for Polyp Segmentation.
Computes:
1. Dice Similarity Coefficient (DSC / F1-Score)
2. Mean Intersection-over-Union (mIoU / Jaccard Index)
3. Precision, Recall (Sensitivity), Specificity
4. Pixel Accuracy (PA)
5. Combined BCE + Dice Loss
6. 4-Panel Diagnostic Segmentation Mask Visualization
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class CombinedBceDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_fn(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def compute_segmentation_metrics(
    preds_prob: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    # preds_prob: [N, H, W] in [0.0, 1.0]
    # targets: [N, H, W] in {0, 1}
    preds_binary = (preds_prob >= threshold).astype(np.uint8)
    targets_binary = (targets >= 0.5).astype(np.uint8)

    intersection = np.sum(preds_binary * targets_binary)
    union = np.sum(np.clip(preds_binary + targets_binary, 0, 1))
    sum_total = np.sum(preds_binary) + np.sum(targets_binary)

    smooth = 1e-6
    dice = (2.0 * intersection + smooth) / (sum_total + smooth)
    iou = (intersection + smooth) / (union + smooth)

    # Pixel confusion matrix
    tp = intersection
    fp = np.sum((preds_binary == 1) & (targets_binary == 0))
    fn = np.sum((preds_binary == 0) & (targets_binary == 1))
    tn = np.sum((preds_binary == 0) & (targets_binary == 0))

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        "Dice_Coefficient": float(dice),
        "mIoU_Jaccard": float(iou),
        "Precision": float(precision),
        "Recall_Sensitivity": float(recall),
        "Specificity": float(specificity),
        "Pixel_Accuracy": float(pixel_acc)
    }


def print_segmentation_report(metrics: Dict[str, Any], model_name: str, seed: Optional[int] = None):
    print("\n" + "="*95)
    print(f"      KVASIR-SEG FOUNDATION SEGMENTATION EVALUATION REPORT: {model_name.upper()}")
    print("="*95)
    if seed is not None:
        print(f" Experiment Seed           : {seed}")
    print(f" Dice Similarity Coeff (DSC): {metrics['Dice_Coefficient']*100:.2f}%")
    print(f" Mean IoU (Jaccard Index)  : {metrics['mIoU_Jaccard']*100:.2f}%")
    print(f" Precision (Positive Pred) : {metrics['Precision']*100:.2f}%")
    print(f" Recall (Lesion Sensitivity): {metrics['Recall_Sensitivity']*100:.2f}%")
    print(f" Specificity (True Negative): {metrics['Specificity']*100:.2f}%")
    print(f" Pixel Accuracy (PA)       : {metrics['Pixel_Accuracy']*100:.2f}%")
    print("="*95 + "\n")


def plot_4panel_segmentation_grid(
    image: Image.Image,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    output_path: Path,
    title: str = "Segmentation Analysis"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_bin = (pred_prob >= 0.50).astype(np.uint8)
    diff = np.abs(pred_bin - gt_mask)

    plt.figure(figsize=(12, 3.5), dpi=300)
    sns.set_theme(style="white")

    # 1. Image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Colonoscopy Frame", weight='bold', fontsize=10)
    plt.axis("off")

    # 2. Ground Truth Mask
    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask, cmap="Blues_r")
    plt.title("Ground Truth Mask", weight='bold', fontsize=10)
    plt.axis("off")

    # 3. Predicted Mask Probability
    plt.subplot(1, 4, 3)
    plt.imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
    plt.title("Predicted Probability", weight='bold', fontsize=10)
    plt.axis("off")

    # 4. Error / Difference Map
    plt.subplot(1, 4, 4)
    plt.imshow(diff, cmap="Reds")
    plt.title("Difference Error Map", weight='bold', fontsize=10)
    plt.axis("off")

    plt.suptitle(title, fontsize=11, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Seg-Metrics] 4-panel visual grid saved to: {output_path}")

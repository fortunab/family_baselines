"""Patch-Based Multi-Resolution Neuroimaging Engine for Alzheimer MRI.

Supports:
1. Downscaling to 32x32 pixels and 8x8 pixels.
2. Patch partitioning:
   - 32x32 partitioned into 4x4 grid of 8x8 patches (16 spatial patches).
   - 8x8 partitioned into 2x2 grid of 4x4 patches (4 macro quadrants) or 64 cellular units.
3. Patch-level descriptor extraction:
   - Patch mean luminance, standard deviation, energy, contrast, edge intensity.
4. Patch-only classification across ML family and evaluation of clinical metrics.
5. Patch saliency mapping (identifying critical ventricular and hippocampal patches).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from src.evaluator import evaluate_clinical_metrics
from src.ml_engine import build_ml_model_catalog
from src.preprocessing import preprocess_mri_slice


def downscale_mri_slice(
    gray_img: np.ndarray,
    target_size: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Downscales a normalized 2D grayscale MRI slice to target resolution using area/bilinear resampling."""
    pil_img = Image.fromarray(np.uint8(np.clip(gray_img, 0.0, 1.0) * 255.0))
    resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.float32) / 255.0


def extract_patches_from_grid(
    img: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Divides a 2D image into non-overlapping spatial patches.

    Returns:
        patches: List of 2D numpy arrays.
        coordinates: List of (row_idx, col_idx) grid positions.
    """
    h, w = img.shape
    ph, pw = patch_size
    num_r = h // ph
    num_c = w // pw

    patches = []
    coordinates = []
    for r in range(num_r):
        for c in range(num_c):
            patch = img[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            patches.append(patch)
            coordinates.append((r, c))

    return patches, coordinates


def extract_patch_descriptors(patch: np.ndarray) -> np.ndarray:
    """Computes dense localized statistical and texture descriptors for a single patch."""
    mean_val = float(np.mean(patch))
    std_val = float(np.std(patch))
    max_val = float(np.max(patch))
    min_val = float(np.min(patch))
    energy = float(np.sum(patch**2))

    # Local gradient within patch
    if patch.shape[0] > 1 and patch.shape[1] > 1:
        dy = np.diff(patch, axis=0)
        dx = np.diff(patch, axis=1)
        grad_mag = float(np.mean(np.abs(dy)) + np.mean(np.abs(dx)))
    else:
        grad_mag = 0.0

    return np.array([mean_val, std_val, max_val, min_val, energy, grad_mag], dtype=np.float32)


def build_patch_only_feature_vector(
    img_32x32: np.ndarray,
    img_8x8: Optional[np.ndarray] = None,
    mode: str = "32x32_patches",
) -> np.ndarray:
    """Builds a localized feature vector looking strictly at patches.

    Modes:
    - '32x32_patches': 16 patches of size 8x8 extracted from 32x32 downscaled slice.
    - '8x8_patches': 4 macro-quadrant patches of size 4x4 (plus raw 64 cells) from 8x8 slice.
    - 'multiscale_fusion': Concatenation of 32x32 patches and 8x8 patches.
    """
    feats = []

    if mode in ["32x32_patches", "multiscale_fusion"]:
        patches_32, _ = extract_patches_from_grid(img_32x32, patch_size=(8, 8))
        for p in patches_32:
            feats.extend(extract_patch_descriptors(p))

    if mode in ["8x8_patches", "multiscale_fusion"]:
        if img_8x8 is None:
            img_8x8 = downscale_mri_slice(img_32x32, target_size=(8, 8))
        # 4 macro quadrants
        quadrants, _ = extract_patches_from_grid(img_8x8, patch_size=(4, 4))
        for q in quadrants:
            feats.extend(extract_patch_descriptors(q))
        # Include normalized raw 8x8 patch cells (64 features)
        feats.extend(img_8x8.flatten())

    return np.array(feats, dtype=np.float32)


def extract_patch_dataset_matrix(
    image_paths: List[str],
    mode: str = "32x32_patches",
) -> np.ndarray:
    """Extracts patch-only feature matrix for a list of MRI image paths."""
    X_list = []
    for p in image_paths:
        res = preprocess_mri_slice(p, target_size=(128, 128))
        gray = res["processed"]
        img_32 = downscale_mri_slice(gray, target_size=(32, 32))
        img_8 = downscale_mri_slice(gray, target_size=(8, 8))

        f_vec = build_patch_only_feature_vector(img_32, img_8, mode=mode)
        X_list.append(f_vec)

    return np.array(X_list, dtype=np.float32)


def compute_patch_saliency_heatmap(
    feature_importances: np.ndarray,
    grid_size: Tuple[int, int] = (4, 4),
    features_per_patch: int = 6,
) -> np.ndarray:
    """Maps feature importance weights back onto the 2D spatial patch grid."""
    num_patches = grid_size[0] * grid_size[1]
    patch_scores = np.zeros(num_patches, dtype=np.float32)

    for i in range(num_patches):
        start_idx = i * features_per_patch
        end_idx = start_idx + features_per_patch
        if end_idx <= len(feature_importances):
            patch_scores[i] = np.sum(feature_importances[start_idx:end_idx])

    heatmap = patch_scores.reshape(grid_size)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

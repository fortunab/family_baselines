"""Grayscale Preprocessing Engine for Alzheimer MRI Scans.

Implements clinical neuroimaging image enhancements:
1. Single-channel luminance conversion (ITU-R BT.601)
2. Spatial noise filtration (Gaussian / Median)
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
4. Parenchymal tissue thresholding and skull-strip masking
5. Percentile dynamic range normalization and spatial resizing
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter


def convert_to_grayscale(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Converts RGB/RGBA image or array to single-channel float32 grayscale in [0, 1]."""
    if isinstance(img, Image.Image):
        gray_pil = img.convert("L")
        return np.array(gray_pil, dtype=np.float32) / 255.0

    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        if arr.shape[-1] >= 3:
            # ITU-R BT.601 standard coefficients
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            return gray.astype(np.float32)
        else:
            return arr[..., 0].astype(np.float32)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {arr.shape}")


def apply_denoising(
    gray_img: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.8,
) -> np.ndarray:
    """Applies noise suppression filter to preserve sulcal and ventricular boundaries."""
    if method == "gaussian":
        return gaussian_filter(gray_img, sigma=sigma)
    elif method == "median":
        return median_filter(gray_img, size=3)
    return gray_img


def apply_clahe_pure_numpy(
    gray_img: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
    num_bins: int = 256,
) -> np.ndarray:
    """Computes Contrast-Limited Adaptive Histogram Equalization (CLAHE) purely in NumPy.

    Partitions image into grid_size x grid_size tiles, computes clipped local histograms,
    equalizes CDFs, and applies bilinear interpolation across tile centers.
    """
    img = np.clip(gray_img, 0.0, 1.0)
    h, w = img.shape
    img_uint8 = np.uint8(img * (num_bins - 1))

    # Tile dimensions
    tile_h = h // grid_size
    tile_w = w // grid_size
    if tile_h == 0 or tile_w == 0:
        return img

    # Compute CDF for each tile
    cdfs = np.zeros((grid_size, grid_size, num_bins), dtype=np.float32)
    clip_thresh = int(clip_limit * (tile_h * tile_w) / num_bins)
    clip_thresh = max(1, clip_thresh)

    for i in range(grid_size):
        r_start = i * tile_h
        r_end = h if i == grid_size - 1 else (i + 1) * tile_h

        for j in range(grid_size):
            c_start = j * tile_w
            c_end = w if j == grid_size - 1 else (j + 1) * tile_w

            tile = img_uint8[r_start:r_end, c_start:c_end]
            hist, _ = np.histogram(tile, bins=num_bins, range=(0, num_bins))

            # Clip excess and redistribute
            excess = np.sum(np.maximum(0, hist - clip_thresh))
            clipped_hist = np.minimum(hist, clip_thresh)
            clipped_hist += excess // num_bins

            # Compute normalized CDF
            cdf = np.cumsum(clipped_hist).astype(np.float32)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
            cdfs[i, j] = cdf

    # Bilinear interpolation across tile grid
    out = np.zeros_like(img, dtype=np.float32)
    y_coords = (np.arange(h) - tile_h / 2.0) / tile_h
    x_coords = (np.arange(w) - tile_w / 2.0) / tile_w

    y1 = np.clip(np.floor(y_coords).astype(int), 0, grid_size - 1)
    y2 = np.clip(y1 + 1, 0, grid_size - 1)
    x1 = np.clip(np.floor(x_coords).astype(int), 0, grid_size - 1)
    x2 = np.clip(x1 + 1, 0, grid_size - 1)

    ay = np.clip(y_coords - y1, 0.0, 1.0)[:, None]  # (h, 1)
    ax = np.clip(x_coords - x1, 0.0, 1.0)[None, :]  # (1, w)

    val = img_uint8
    f00 = cdfs[y1[:, None], x1[None, :], val]
    f01 = cdfs[y1[:, None], x2[None, :], val]
    f10 = cdfs[y2[:, None], x1[None, :], val]
    f11 = cdfs[y2[:, None], x2[None, :], val]

    top = f00 * (1.0 - ax) + f01 * ax
    bot = f10 * (1.0 - ax) + f11 * ax
    out = top * (1.0 - ay) + bot * ay

    return np.clip(out, 0.0, 1.0)


def extract_brain_mask(
    gray_img: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Computes binary mask isolating brain parenchyma from dark scanner background."""
    norm = (gray_img - np.min(gray_img)) / (np.max(gray_img) - np.min(gray_img) + 1e-8)
    return (norm > threshold).astype(np.float32)


def preprocess_mri_slice(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (128, 128),
    denoise_method: str = "gaussian",
    sigma: float = 0.8,
    enable_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: int = 8,
    mask_background: bool = True,
    bg_threshold: float = 0.04,
) -> Dict[str, np.ndarray]:
    """Applies end-to-end clinical grayscale preprocessing pipeline to an MRI slice.

    Returns dictionary containing:
    - raw_gray: Raw luminance slice resized to target size
    - denoised: Noise-suppressed slice
    - clahe_enhanced: Contrast-enhanced slice
    - brain_mask: Binary parenchyma mask
    - processed: Final preprocessed normalized grayscale array in [0, 1]
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(np.uint8(image_input * 255.0) if image_input.max() <= 1.0 else np.uint8(image_input))
    else:
        pil_img = image_input

    # 1. Resize and convert to grayscale
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    raw_gray = convert_to_grayscale(pil_resized)

    # 2. Denoising
    denoised = apply_denoising(raw_gray, method=denoise_method, sigma=sigma)

    # 3. CLAHE Contrast enhancement
    enhanced = (
        apply_clahe_pure_numpy(denoised, clip_limit=clahe_clip_limit, grid_size=clahe_grid_size)
        if enable_clahe
        else denoised
    )

    # 4. Parenchyma masking & normalization
    mask = extract_brain_mask(enhanced, threshold=bg_threshold) if mask_background else np.ones_like(enhanced)
    final_processed = enhanced * mask

    # Percentile clipping
    p1, p99 = np.percentile(final_processed[mask > 0] if np.any(mask > 0) else final_processed, (1.0, 99.0))
    if p99 > p1:
        final_processed = np.clip((final_processed - p1) / (p99 - p1), 0.0, 1.0)
    final_processed = final_processed * mask

    return {
        "raw_gray": raw_gray,
        "denoised": denoised,
        "clahe_enhanced": enhanced,
        "brain_mask": mask,
        "processed": final_processed,
    }


def batch_preprocess_dataset(
    raw_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int] = (128, 128),
) -> int:
    """Preprocesses entire directory of MRI scans and saves them as 8-bit PNG grayscale images."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    count = 0

    for cls_dir in raw_path.iterdir():
        if cls_dir.is_dir():
            target_cls_dir = out_path / cls_dir.name
            target_cls_dir.mkdir(parents=True, exist_ok=True)

            for img_f in cls_dir.iterdir():
                if img_f.is_file() and img_f.suffix.lower() in valid_exts:
                    res = preprocess_mri_slice(img_f, target_size=target_size)
                    out_img_arr = np.uint8(res["processed"] * 255.0)
                    out_pil = Image.fromarray(out_img_arr, mode="L")
                    out_file = target_cls_dir / f"{img_f.stem}_preproc.png"
                    out_pil.save(out_file)
                    count += 1

    return count

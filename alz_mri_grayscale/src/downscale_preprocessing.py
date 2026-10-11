"""Downscaling and Grayscale Preprocessing Pipeline for Alzheimer MRI.

Implements the user-specified preprocessing pipeline:
1. Spatial Downscaling:
   - Context A: Downscaling to 32x32 pixels
   - Context B: Downscaling to 8x8 pixels
2. Followed by Grayscaling (in both contexts):
   - Luminance conversion (ITU-R BT.601) to single-channel [0, 1] float32 and [0, 255] uint8 PNG.
3. Information-theoretic and structural quality evaluation:
   - Shannon Entropy, Laplacian edge energy, PSNR, and SSIM of reconstructed slices.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image


def compute_shannon_entropy(gray_arr: np.ndarray, num_bins: int = 256) -> float:
    """Computes Shannon entropy H = -sum(p * log2(p)) over normalized luminance histogram."""
    hist, _ = np.histogram(gray_arr, bins=num_bins, range=(0.0, 1.0))
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    return round(entropy, 4)


def compute_laplacian_variance(gray_arr: np.ndarray) -> float:
    """Computes Laplacian second-derivative variance as a proxy for high-frequency edge energy."""
    # Simple discrete 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray_arr.shape
    if h < 3 or w < 3:
        return 0.0

    # Padded convolution
    pad = np.pad(gray_arr, 1, mode="reflect")
    lap = np.zeros_like(gray_arr)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * pad[i : i + h, j : j + w]

    var_lap = float(np.var(lap))
    return round(var_lap, 6)


def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed array."""
    mse = float(np.mean((orig - recon) ** 2))
    if mse < 1e-10:
        return 99.0
    psnr = 10.0 * np.log10(1.0 / mse)
    return round(float(psnr), 2)


def compute_ssim_numpy(
    x: np.ndarray,
    y: np.ndarray,
    c1: float = (0.01) ** 2,
    c2: float = (0.03) ** 2,
) -> float:
    """Computes structural similarity index (SSIM) purely in NumPy."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x_sq = np.var(x)
    sigma_y_sq = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim = float(numerator / denominator)
    return round(float(ssim), 4)


def preprocess_slice_downscale_then_gray(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.LANCZOS,
) -> Dict[str, Any]:
    """Applies downscaling first, followed by grayscaling.

    Parameters:
        image_input: Filepath, PIL Image, or NumPy array of original MRI scan.
        target_size: Target downscaled spatial resolution (e.g., (32, 32) or (8, 8)).
        resampling: High-order anti-aliasing resampling filter.

    Returns:
        Dict with downscaled PIL image, normalized float32 grayscale array, uint8 array,
        and statistical descriptors.
    """
    if isinstance(image_input, (str, Path)):
        pil_orig = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.max() <= 1.0:
            pil_orig = Image.fromarray(np.uint8(image_input * 255.0))
        else:
            pil_orig = Image.fromarray(np.uint8(image_input))
    elif isinstance(image_input, Image.Image):
        pil_orig = image_input
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    orig_w, orig_h = pil_orig.size

    # STEP 1: DOWNSCALING as part of preprocessing
    pil_downscaled = pil_orig.resize(target_size, resample=resampling)

    # STEP 2: GRAYSCALING in this context
    pil_gray = pil_downscaled.convert("L")
    arr_uint8 = np.array(pil_gray, dtype=np.uint8)
    arr_float = arr_uint8.astype(np.float32) / 255.0

    # Also compute reconstructed slice to evaluate fidelity loss relative to original
    pil_recon = pil_gray.resize((orig_w, orig_h), resample=Image.Resampling.BICUBIC)
    arr_recon = np.array(pil_recon, dtype=np.float32) / 255.0

    orig_gray = np.array(pil_orig.convert("L"), dtype=np.float32) / 255.0
    ssim_val = compute_ssim_numpy(orig_gray, arr_recon)
    psnr_val = compute_psnr(orig_gray, arr_recon)
    entropy_val = compute_shannon_entropy(arr_float)
    edge_energy = compute_laplacian_variance(arr_float)

    return {
        "target_size": target_size,
        "pil_gray": pil_gray,
        "arr_float": arr_float,
        "arr_uint8": arr_uint8,
        "arr_recon": arr_recon,
        "orig_gray": orig_gray,
        "metrics": {
            "mean": round(float(np.mean(arr_float)), 4),
            "std": round(float(np.std(arr_float)), 4),
            "min": round(float(np.min(arr_float)), 4),
            "max": round(float(np.max(arr_float)), 4),
            "entropy": entropy_val,
            "edge_energy": edge_energy,
            "reconstruction_ssim": ssim_val,
            "reconstruction_psnr": psnr_val,
        },
    }


def execute_batch_downscaling(
    raw_dir: Union[str, Path],
    dir_32: Union[str, Path],
    dir_8: Union[str, Path],
    manifest_output_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Processes entire dataset of Alzheimer MRI scans into 32x32 and 8x8 grayscaled datasets."""
    raw_p = Path(raw_dir).resolve()
    p_32 = Path(dir_32).resolve()
    p_8 = Path(dir_8).resolve()

    p_32.mkdir(parents=True, exist_ok=True)
    p_8.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files: List[Path] = []
    for f in raw_p.rglob("*"):
        if f.is_file() and f.suffix.lower() in valid_exts:
            files.append(f)

    files = sorted(files)
    records = []

    for f in files:
        cls_name = f.parent.name
        sample_id = f.stem

        out_cls_32 = p_32 / cls_name
        out_cls_8 = p_8 / cls_name
        out_cls_32.mkdir(parents=True, exist_ok=True)
        out_cls_8.mkdir(parents=True, exist_ok=True)

        # Context 1: 32x32
        res_32 = preprocess_slice_downscale_then_gray(f, target_size=(32, 32))
        file_32 = out_cls_32 / f"{sample_id}_32x32.png"
        res_32["pil_gray"].save(file_32)

        # Context 2: 8x8
        res_8 = preprocess_slice_downscale_then_gray(f, target_size=(8, 8))
        file_8 = out_cls_8 / f"{sample_id}_8x8.png"
        res_8["pil_gray"].save(file_8)

        m32 = res_32["metrics"]
        m8 = res_8["metrics"]

        records.append(
            {
                "sample_id": sample_id,
                "class_name": cls_name,
                "original_path": str(f).replace("\\", "/"),
                "path_32x32": str(file_32).replace("\\", "/"),
                "path_8x8": str(file_8).replace("\\", "/"),
                "entropy_32x32": m32["entropy"],
                "entropy_8x8": m8["entropy"],
                "edge_energy_32x32": m32["edge_energy"],
                "edge_energy_8x8": m8["edge_energy"],
                "ssim_recon_32x32": m32["reconstruction_ssim"],
                "ssim_recon_8x8": m8["reconstruction_ssim"],
                "psnr_recon_32x32": m32["reconstruction_psnr"],
                "psnr_recon_8x8": m8["reconstruction_psnr"],
                "mean_32x32": m32["mean"],
                "mean_8x8": m8["mean"],
            }
        )

    df = pd.DataFrame(records)

    if manifest_output_path:
        m_p = Path(manifest_output_path)
        m_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(m_p, index=False)

    summary = {
        "total_scans_processed": len(df),
        "classes_discovered": sorted(list(df["class_name"].unique())),
        "compression_ratios": {
            "original_size": [176, 208],
            "original_pixels": 36608,
            "context_32x32": {
                "dimensions": [32, 32],
                "pixels": 1024,
                "compression_ratio": round(36608 / 1024, 2),
                "data_reduction_percent": round((1.0 - 1024 / 36608) * 100, 2),
                "avg_recon_ssim": round(float(df["ssim_recon_32x32"].mean()), 4),
                "avg_recon_psnr": round(float(df["psnr_recon_32x32"].mean()), 2),
                "avg_entropy": round(float(df["entropy_32x32"].mean()), 4),
            },
            "context_8x8": {
                "dimensions": [8, 8],
                "pixels": 64,
                "compression_ratio": round(36608 / 64, 2),
                "data_reduction_percent": round((1.0 - 64 / 36608) * 100, 2),
                "avg_recon_ssim": round(float(df["ssim_recon_8x8"].mean()), 4),
                "avg_recon_psnr": round(float(df["psnr_recon_8x8"].mean()), 2),
                "avg_entropy": round(float(df["entropy_8x8"].mean()), 4),
            },
        },
    }

    return df, summary

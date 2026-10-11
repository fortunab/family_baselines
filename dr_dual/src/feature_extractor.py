"""Radiomic and Clinical Biomarker Feature Extraction Engine (Pure NumPy).

Extracts 42 clinical biomarker and texture features from retinal fundus photographs:
1. Green-channel optical contrast (hemoglobin absorption peak)
2. Retinal blood vessel density and morphological tortuosity proxies
3. Optic disc localization and cup-to-disc contrast
4. Dark lesion radiomics (microaneurysms & dot-blot hemorrhages)
5. Bright lesion radiomics (hard exudates & cotton-wool spots)
6. Pure-NumPy Gray-Level Co-occurrence Matrix (GLCM) texture moments
7. Multi-color space statistical moments (RGB, CIE L*a*b*, HSV)
8. Pure-NumPy Local Binary Patterns (LBP) micro-texture descriptors
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    label,
    morphological_gradient,
)


def compute_pure_numpy_glcm_props(gray_quant: np.ndarray, levels: int = 16) -> Dict[str, float]:
    """Computes GLCM properties using pure NumPy without external C-extensions."""
    # horizontal pairs (i, j)
    h_pairs_1 = gray_quant[:, :-1].ravel()
    h_pairs_2 = gray_quant[:, 1:].ravel()
    # vertical pairs (i, j)
    v_pairs_1 = gray_quant[:-1, :].ravel()
    v_pairs_2 = gray_quant[1:, :].ravel()

    i_vals = np.concatenate([h_pairs_1, v_pairs_1])
    j_vals = np.concatenate([h_pairs_2, v_pairs_2])

    flat_idx = i_vals.astype(np.int64) * levels + j_vals.astype(np.int64)
    glcm = np.bincount(flat_idx, minlength=levels * levels).reshape((levels, levels)).astype(np.float64)
    glcm = glcm + glcm.T
    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        glcm /= glcm_sum

    i_grid, j_grid = np.mgrid[0:levels, 0:levels]
    diff = i_grid - j_grid
    diff_abs = np.abs(diff)
    diff_sq = diff**2

    contrast = float(np.sum(glcm * diff_sq))
    dissimilarity = float(np.sum(glcm * diff_abs))
    homogeneity = float(np.sum(glcm / (1.0 + diff_sq)))
    energy = float(np.sqrt(np.sum(glcm**2)))
    asm = float(np.sum(glcm**2))

    i_mean = np.sum(i_grid * glcm)
    j_mean = np.sum(j_grid * glcm)
    i_var = np.sum(((i_grid - i_mean) ** 2) * glcm)
    j_var = np.sum(((j_grid - j_mean) ** 2) * glcm)
    denom = np.sqrt(i_var * j_var) + 1e-9
    correlation = float(np.sum((i_grid - i_mean) * (j_grid - j_mean) * glcm) / denom)

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
        "asm": asm,
    }


def compute_pure_numpy_lbp_stats(gray_quant: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Computes 8-neighbor Local Binary Pattern stats using vectorized NumPy slice operations."""
    center = gray_quant[1:-1, 1:-1]
    neighbors = [
        gray_quant[:-2, :-2],
        gray_quant[:-2, 1:-1],
        gray_quant[:-2, 2:],
        gray_quant[1:-1, 2:],
        gray_quant[2:, 2:],
        gray_quant[2:, 1:-1],
        gray_quant[2:, :-2],
        gray_quant[1:-1, :-2],
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit_idx, n_img in enumerate(neighbors):
        lbp += (n_img >= center).astype(np.uint8) << bit_idx

    if mask is not None:
        inner_mask = mask[1:-1, 1:-1]
        valid_vals = lbp[inner_mask]
    else:
        valid_vals = lbp.flatten()

    if len(valid_vals) == 0:
        return 0.0, 0.0

    hist, _ = np.histogram(valid_vals, bins=16, range=(0, 256), density=True)
    uniformity = float(np.sum(hist**2))
    entropy = -float(np.sum(hist * np.log(np.clip(hist, 1e-7, 1.0))))
    return uniformity, entropy


def extract_fundus_biomarker_features(
    image_input: Union[str, Path, np.ndarray],
    retina_crop_size: int = 256,
) -> Tuple[np.ndarray, List[str]]:
    """Extracts a 42-dimensional clinical feature vector from a retinal fundus scan."""
    if isinstance(image_input, (str, Path)):
        img_pil = Image.open(image_input).convert("RGB").resize((retina_crop_size, retina_crop_size))
        img = np.array(img_pil)
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if img.shape[:2] != (retina_crop_size, retina_crop_size):
            img = np.array(Image.fromarray(img).resize((retina_crop_size, retina_crop_size)))
    else:
        raise TypeError(f"Unsupported image input: {type(image_input)}")

    h, w, _ = img.shape
    features: List[float] = []
    names: List[str] = []

    # 1. Retinal Mask Extraction (Exclude black background)
    gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
    retina_mask = gray > 18.0
    retina_pixels = max(1, int(np.sum(retina_mask)))

    features.append(float(retina_pixels / (h * w)))
    names.append("retina_area_ratio")

    # 2. Green Channel Analysis (Highest hemoglobin contrast)
    green = img[:, :, 1].astype(np.float32)
    green_fg = green[retina_mask] if np.any(retina_mask) else green.flatten()

    g_mean = float(np.mean(green_fg))
    g_std = float(np.std(green_fg))
    g_skew = float(np.mean(((green_fg - g_mean) / (g_std + 1e-6)) ** 3))

    features.extend([g_mean, g_std, g_skew])
    names.extend(["green_mean", "green_std", "green_skewness"])

    # 3. Retinal Blood Vessel Segmentation Proxy
    smooth_green = gaussian_filter(green, sigma=1.0)
    vessel_diff = smooth_green - gaussian_filter(green, sigma=4.0)
    vessel_mask = (vessel_diff < -6.0) & retina_mask
    vessel_pixel_count = int(np.sum(vessel_mask))
    vessel_density = float(vessel_pixel_count / retina_pixels)

    vessel_grad = morphological_gradient(vessel_mask.astype(np.float32), size=3)
    vessel_edge_count = float(np.sum(vessel_grad > 0))
    vessel_tortuosity = float(vessel_edge_count / max(1, vessel_pixel_count))

    features.extend([vessel_density, vessel_tortuosity])
    names.extend(["vessel_density", "vessel_tortuosity_proxy"])

    # 4. Optic Disc Localization & Contrast Proxy
    red = img[:, :, 0].astype(np.float32)
    disc_filtered = gaussian_filter(red, sigma=8.0)
    disc_filtered[~retina_mask] = 0.0
    disc_py, disc_px = np.unravel_index(np.argmax(disc_filtered), disc_filtered.shape)

    y_g, x_g = np.mgrid[0:h, 0:w]
    disc_mask = ((x_g - disc_px) ** 2 + (y_g - disc_py) ** 2) <= (22**2)
    disc_intensity = float(np.mean(red[disc_mask & retina_mask])) if np.any(disc_mask & retina_mask) else g_mean
    disc_contrast = disc_intensity / (float(np.mean(red[retina_mask])) + 1e-6)

    features.extend([disc_contrast, float(disc_px / w), float(disc_py / h)])
    names.extend(["optic_disc_contrast", "optic_disc_pos_x", "optic_disc_pos_y"])

    # 5. Hemorrhage and Microaneurysm Lesion Candidates (Dark Lesions)
    dark_lesion_candidates = (vessel_diff < -15.0) & retina_mask & (~disc_mask)
    dark_lesion_area = float(np.sum(dark_lesion_candidates) / retina_pixels)
    labeled_dark, num_dark_clusters = label(dark_lesion_candidates)

    features.extend([dark_lesion_area, float(num_dark_clusters)])
    names.extend(["dark_lesion_area_ratio", "dark_lesion_cluster_count"])

    # 6. Hard Exudate and Cotton-Wool Spot Candidates (Bright Lesions)
    bright_diff = green - gaussian_filter(green, sigma=12.0)
    bright_lesion_candidates = (bright_diff > 20.0) & retina_mask & (~disc_mask)
    bright_lesion_area = float(np.sum(bright_lesion_candidates) / retina_pixels)
    labeled_bright, num_bright_clusters = label(bright_lesion_candidates)

    features.extend([bright_lesion_area, float(num_bright_clusters)])
    names.extend(["bright_lesion_area_ratio", "bright_lesion_cluster_count"])

    # 7. Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors (Pure NumPy)
    gray_quant = np.clip((gray / 16.0).astype(np.uint8), 0, 15)
    glcm_dict = compute_pure_numpy_glcm_props(gray_quant, levels=16)

    features.extend(
        [
            glcm_dict["contrast"],
            glcm_dict["dissimilarity"],
            glcm_dict["homogeneity"],
            glcm_dict["energy"],
            glcm_dict["correlation"],
            glcm_dict["asm"],
        ]
    )
    names.extend(
        [
            "glcm_contrast",
            "glcm_dissimilarity",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "glcm_asm",
        ]
    )

    # 8. Multi-Color Space Statistical Moments
    for c_i, c_name in enumerate(["red", "green", "blue"]):
        ch = img[:, :, c_i].astype(np.float32)
        ch_fg = ch[retina_mask] if np.any(retina_mask) else ch.flatten()
        m = float(np.mean(ch_fg))
        s = float(np.std(ch_fg))
        k = float(np.mean(((ch_fg - m) / (s + 1e-6)) ** 3))
        features.extend([m, s, k])
        names.extend([f"{c_name}_mean", f"{c_name}_std", f"{c_name}_skewness"])

    lab_l = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(np.float32)
    lab_a = img[:, :, 0].astype(np.float32) - img[:, :, 1].astype(np.float32)
    lab_b = img[:, :, 1].astype(np.float32) - img[:, :, 2].astype(np.float32)

    for ch, c_name in [(lab_l, "lab_L"), (lab_a, "lab_a"), (lab_b, "lab_b")]:
        ch_fg = ch[retina_mask] if np.any(retina_mask) else ch.flatten()
        m = float(np.mean(ch_fg))
        s = float(np.std(ch_fg))
        features.extend([m, s])
        names.extend([f"{c_name}_mean", f"{c_name}_std"])

    max_c = np.max(img, axis=-1).astype(np.float32)
    min_c = np.min(img, axis=-1).astype(np.float32)
    sat = np.zeros_like(max_c)
    non_zero = max_c > 0
    sat[non_zero] = (max_c[non_zero] - min_c[non_zero]) / max_c[non_zero]

    val_fg = max_c[retina_mask] if np.any(retina_mask) else max_c.flatten()
    sat_fg = sat[retina_mask] if np.any(retina_mask) else sat.flatten()

    features.extend([float(np.mean(val_fg)), float(np.std(val_fg)), float(np.mean(sat_fg)), float(np.std(sat_fg))])
    names.extend(["hsv_value_mean", "hsv_value_std", "hsv_saturation_mean", "hsv_saturation_std"])

    # 9. Local Binary Patterns (LBP) Micro-Texture (Pure NumPy)
    lbp_uniformity, lbp_entropy = compute_pure_numpy_lbp_stats(gray_quant, retina_mask)
    features.extend([lbp_uniformity, lbp_entropy])
    names.extend(["lbp_uniformity", "lbp_entropy"])

    return np.array(features, dtype=np.float32), names


def extract_features_for_dataset(
    samples: List[Any],
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """Extracts radiomic feature matrix X and target labels y for a list of samples."""
    x_list = []
    y_list = []
    metadata = []
    feature_names = []

    for idx, s in enumerate(samples):
        vec, names = extract_fundus_biomarker_features(s.image_path)
        if not feature_names:
            feature_names = names

        x_list.append(vec)
        y_list.append(s.label_idx)
        metadata.append(
            {
                "image_id": s.image_id,
                "database": s.database,
                "split": s.split,
                "label_idx": s.label_idx,
                "label_name": s.label_name,
                "is_referable": s.is_referable,
            }
        )

    X = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y, feature_names, metadata

"""
Feature Extraction Pipeline for Colorectal Histology Baseline.
Implements handcrafted texture & color descriptors as defined in Kather et al. (2016):
1. Multiscale Local Binary Patterns (LBP)
2. Gray-Level Co-occurrence Matrix (GLCM / Haralick features)
3. Gabor Filter Bank (Multi-scale, Multi-orientation energy & moments)
4. Color Features (RGB, HSV, CIE-Lab statistical moments and histograms)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import stats
from scipy import ndimage
from skimage import color
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


class CombinedFeatureExtractor:
    """
    Extracts LBP, GLCM, Gabor, and Color features from histology RGB image patches.
    """

    def __init__(
        self,
        lbp_radii_points: List[Tuple[int, int]] = [(1, 8), (2, 16), (3, 24)],
        glcm_distances: List[int] = [1, 2, 3, 5],
        glcm_angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        gabor_frequencies: List[float] = [0.05, 0.1, 0.2, 0.35],
        gabor_orientations: List[float] = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6],
        include_lbp: bool = True,
        include_glcm: bool = True,
        include_gabor: bool = True,
        include_color: bool = True
    ):
        self.lbp_radii_points = lbp_radii_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.gabor_frequencies = gabor_frequencies
        self.gabor_orientations = gabor_orientations
        
        self.include_lbp = include_lbp
        self.include_glcm = include_glcm
        self.include_gabor = include_gabor
        self.include_color = include_color

        # Precompute Gabor kernels for fast convolution
        self.gabor_kernels = []
        for freq in self.gabor_frequencies:
            for theta in self.gabor_orientations:
                # Real and Imaginary Gabor kernels
                kernel_real = np.real(gabor_kernel(freq, theta=theta))
                kernel_imag = np.imag(gabor_kernel(freq, theta=theta))
                self.gabor_kernels.append((kernel_real, kernel_imag))

    def extract_lbp_features(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Extracts multi-scale uniform Local Binary Pattern histograms.
        gray_img: 2D uint8 array (H, W)
        """
        lbp_feats = []
        for radius, n_points in self.lbp_radii_points:
            # 'uniform' LBP creates n_points + 2 output bin categories
            lbp = local_binary_pattern(gray_img, P=n_points, R=radius, method='uniform')
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            lbp_feats.append(hist)
        return np.concatenate(lbp_feats)

    def extract_glcm_features(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Extracts Haralick texture features from Gray-Level Co-occurrence Matrix.
        gray_img: 2D uint8 array (H, W)
        """
        # Quantize to 64 levels for robust co-occurrence statistics
        gray_quantized = (gray_img // 4).astype(np.uint8)
        
        glcm = graycomatrix(
            gray_quantized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=64,
            symmetric=True,
            normed=True
        )

        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        feats = []
        for prop in properties:
            val = graycoprops(glcm, prop)  # Shape: (len(distances), len(angles))
            # Include all distance-angle combinations
            feats.append(val.ravel())
            # Also include aggregate mean and standard deviation across angles
            feats.append(np.mean(val, axis=1))
            feats.append(np.std(val, axis=1))

        return np.concatenate(feats)

    def extract_gabor_features(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Extracts mean response and energy across a 2D Gabor filter bank.
        gray_img: 2D float array (H, W) in [0, 1]
        """
        gabor_feats = []
        img_float = gray_img.astype(np.float64) / 255.0

        for kernel_real, kernel_imag in self.gabor_kernels:
            # 2D spatial convolution
            filtered_real = ndimage.convolve(img_float, kernel_real, mode='reflect')
            filtered_imag = ndimage.convolve(img_float, kernel_imag, mode='reflect')
            
            # Response magnitude
            magnitude = np.hypot(filtered_real, filtered_imag)
            
            # First and second order energy moments
            mean_val = np.mean(magnitude)
            std_val = np.std(magnitude)
            energy_val = np.mean(magnitude ** 2)
            
            gabor_feats.extend([mean_val, std_val, energy_val])

        return np.array(gabor_feats, dtype=np.float64)

    def extract_color_features(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Extracts statistical moments (mean, std, skewness) in RGB, HSV, and CIE-Lab spaces,
        plus color histograms.
        rgb_img: 3D uint8 array (H, W, 3)
        """
        color_feats = []
        
        # 1. RGB Moments
        rgb_float = rgb_img.astype(np.float64) / 255.0
        for c in range(3):
            ch = rgb_float[:, :, c].ravel()
            mean_val = np.mean(ch)
            std_val = np.std(ch)
            skew_val = stats.skew(ch)
            color_feats.extend([mean_val, std_val, skew_val])

        # 2. HSV Moments & Histograms
        hsv_img = color.rgb2hsv(rgb_img)
        for c in range(3):
            ch = hsv_img[:, :, c].ravel()
            mean_val = np.mean(ch)
            std_val = np.std(ch)
            skew_val = stats.skew(ch)
            color_feats.extend([mean_val, std_val, skew_val])
            
        # Hue & Saturation histograms
        hue_hist, _ = np.histogram(hsv_img[:, :, 0].ravel(), bins=16, range=(0, 1), density=True)
        sat_hist, _ = np.histogram(hsv_img[:, :, 1].ravel(), bins=8, range=(0, 1), density=True)
        color_feats.extend(hue_hist)
        color_feats.extend(sat_hist)

        # 3. CIE-Lab Moments
        lab_img = color.rgb2lab(rgb_img)
        for c in range(3):
            ch = lab_img[:, :, c].ravel()
            mean_val = np.mean(ch)
            std_val = np.std(ch)
            skew_val = stats.skew(ch)
            color_feats.extend([mean_val, std_val, skew_val])

        # 4. RGB 8-bin marginal histograms
        for c in range(3):
            ch_hist, _ = np.histogram(rgb_float[:, :, c].ravel(), bins=8, range=(0, 1), density=True)
            color_feats.extend(ch_hist)

        return np.array(color_feats, dtype=np.float64)

    def extract_single_image(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Extracts and concatenates all requested descriptors for a single RGB image.
        Returns a 1D float64 feature vector.
        """
        # Grayscale conversion for texture extraction
        if rgb_img.ndim == 3:
            gray_img = (color.rgb2gray(rgb_img) * 255.0).astype(np.uint8)
        else:
            gray_img = rgb_img
            rgb_img = np.stack([rgb_img]*3, axis=-1)

        feature_blocks = []

        if self.include_lbp:
            lbp_v = self.extract_lbp_features(gray_img)
            feature_blocks.append(lbp_v)

        if self.include_glcm:
            glcm_v = self.extract_glcm_features(gray_img)
            feature_blocks.append(glcm_v)

        if self.include_gabor:
            gabor_v = self.extract_gabor_features(gray_img)
            feature_blocks.append(gabor_v)

        if self.include_color:
            color_v = self.extract_color_features(rgb_img)
            feature_blocks.append(color_v)

        return np.concatenate(feature_blocks).astype(np.float64)


def extract_features_parallel(
    image_paths: List[str],
    labels: np.ndarray,
    extractor: Optional[CombinedFeatureExtractor] = None,
    n_jobs: int = -1,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features for all images in parallel, with automatic caching to .npz file.
    Returns (X, y) where X is (N, D) and y is (N,).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            print(f"[FeatureExtractor] Loading cached features from {cache_path}...")
            data = np.load(cache_path)
            X = data["X"]
            y = data["y"]
            print(f"[FeatureExtractor] Loaded features shape: {X.shape}, labels shape: {y.shape}")
            return X, y

    if extractor is None:
        extractor = CombinedFeatureExtractor()

    print(f"[FeatureExtractor] Starting parallel feature extraction on {len(image_paths)} images using {n_jobs} jobs...")

    def process_path(path: str) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        return extractor.extract_single_image(rgb)

    results = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(process_path)(p) for p in tqdm(image_paths, desc="Extracting Descriptors")
    )

    X = np.vstack(results)
    y = np.array(labels, dtype=np.int64)

    # Clean any NaN / Inf values if present
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[FeatureExtractor] Extraction complete! Feature matrix shape: {X.shape}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FeatureExtractor] Caching extracted features to {cache_path}...")
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


if __name__ == "__main__":
    # Quick sanity test on synthetic image
    test_img = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
    ext = CombinedFeatureExtractor()
    feat = ext.extract_single_image(test_img)
    print(f"Sanity test passed! Extracted feature vector dimension: {feat.shape[0]}")
    print(f"LBP dimension: {ext.extract_lbp_features((test_img[:,:,0])).shape[0]}")
    print(f"GLCM dimension: {ext.extract_glcm_features((test_img[:,:,0])).shape[0]}")
    print(f"Gabor dimension: {ext.extract_gabor_features((test_img[:,:,0])).shape[0]}")
    print(f"Color dimension: {ext.extract_color_features(test_img).shape[0]}")

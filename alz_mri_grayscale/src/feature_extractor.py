"""Feature Extraction Engine for Classical ML Family on Grayscaled Alzheimer MRI.

Extracts:
1. Radiomic intensity moments & entropy
2. Radial brain profile features (ventricular dilation vs cortical thinning)
3. GLCM-inspired texture features (contrast, dissimilarity, homogeneity, energy)
4. HOG gradient orientations & PCA spectral reduction
5. Pretrained Foundation feature embeddings (DenseNet / ResNet / ConvNeXt)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import sobel
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

from src.preprocessing import preprocess_mri_slice


def extract_radiomic_moments(gray_img: np.ndarray) -> np.ndarray:
    """Extracts first-order statistical moments and percentile distributions of brain tissue."""
    vals = gray_img[gray_img > 0.02]
    if len(vals) < 10:
        vals = gray_img.flatten()

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    var_val = float(np.var(vals))
    skew_val = float(skew(vals))
    kurt_val = float(kurtosis(vals))

    p = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])

    # Shannon entropy of intensity histogram
    hist, _ = np.histogram(vals, bins=64, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    entropy_val = float(-np.sum(hist * np.log2(hist + 1e-12)))

    features = [mean_val, std_val, var_val, skew_val, kurt_val, entropy_val] + list(p)
    return np.array(features, dtype=np.float32)


def extract_radial_atrophy_profile(gray_img: np.ndarray, num_rings: int = 8) -> np.ndarray:
    """Extracts concentric radial intensity profiles from the center of the brain outwards.

    Directly quantifies central ventricular dilation vs peripheral cortical thinning.
    """
    h, w = gray_img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)

    ring_means = []
    ring_stds = []
    for i in range(num_rings):
        r_inner = (i / num_rings) * max_r
        r_outer = ((i + 1) / num_rings) * max_r
        mask = (r >= r_inner) & (r < r_outer) & (gray_img > 0.02)
        if np.any(mask):
            ring_means.append(float(np.mean(gray_img[mask])))
            ring_stds.append(float(np.std(gray_img[mask])))
        else:
            ring_means.append(0.0)
            ring_stds.append(0.0)

    return np.array(ring_means + ring_stds, dtype=np.float32)


def extract_glcm_texture_features(gray_img: np.ndarray) -> np.ndarray:
    """Extracts Haralick-style Gray-Level Co-occurrence Matrix (GLCM) texture metrics.

    Computes Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
    """
    # Downscale intensity to 16 gray levels for rapid GLCM computation
    quantized = np.clip(np.uint8(gray_img * 15), 0, 15)
    features = []

    # Offsets: horizontal (0, 1) and vertical (1, 0)
    offsets = [(0, 1), (1, 0), (1, 1)]
    for dy, dx in offsets:
        h, w = quantized.shape
        y_max = h - dy if dy > 0 else h
        x_max = w - dx if dx > 0 else w
        y_min = 0 if dy >= 0 else -dy
        x_min = 0 if dx >= 0 else -dx

        i_indices = quantized[y_min:y_max, x_min:x_max].flatten()
        j_indices = quantized[y_min + dy : y_max + dy, x_min + dx : x_max + dx].flatten()

        matrix, _, _ = np.histogram2d(i_indices, j_indices, bins=16, range=[[0, 16], [0, 16]])
        matrix = matrix / (np.sum(matrix) + 1e-8)

        i_grid, j_grid = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        diff = np.abs(i_grid - j_grid)

        contrast = float(np.sum(matrix * (diff**2)))
        dissimilarity = float(np.sum(matrix * diff))
        homogeneity = float(np.sum(matrix / (1.0 + diff**2)))
        energy = float(np.sum(matrix**2))

        features.extend([contrast, dissimilarity, homogeneity, energy])

    return np.array(features, dtype=np.float32)


def extract_gradient_edge_features(gray_img: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Computes directional edge gradient magnitudes and orientation histograms."""
    gx = sobel(gray_img, axis=1)
    gy = sobel(gray_img, axis=0)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)  # [-pi, pi]
    orientation = (orientation + np.pi) / (2.0 * np.pi)  # [0, 1]

    hist, _ = np.histogram(orientation, bins=num_bins, range=(0.0, 1.0), weights=magnitude)
    hist_norm = hist / (np.sum(hist) + 1e-8)

    mag_mean = float(np.mean(magnitude))
    mag_std = float(np.std(magnitude))

    return np.array(list(hist_norm) + [mag_mean, mag_std], dtype=np.float32)


def extract_handcrafted_features_single(gray_img: np.ndarray) -> np.ndarray:
    """Combines Radiomic moments, radial atrophy profiles, GLCM texture, and gradient edges."""
    f_moments = extract_radiomic_moments(gray_img)
    f_radial = extract_radial_atrophy_profile(gray_img, num_rings=8)
    f_texture = extract_glcm_texture_features(gray_img)
    f_edges = extract_gradient_edge_features(gray_img, num_bins=8)

    return np.concatenate([f_moments, f_radial, f_texture, f_edges])


class FoundationFeatureExtractor:
    """Extracts deep semantic embeddings from pretrained vision foundation backbones."""

    def __init__(self, backbone_name: str = "resnet50", device: str = "cpu"):
        self.device = device
        self.backbone_name = backbone_name

        if backbone_name == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.encoder = model.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
            self.pool = torch.nn.Identity()
        elif backbone_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.encoder = model.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
            self.pool = torch.nn.Identity()

        self.encoder.to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_from_grayscale(self, gray_img: np.ndarray) -> np.ndarray:
        """Takes single-channel float32 grayscale image, replicates to 3 channels, and extracts embedding."""
        h, w = gray_img.shape
        if (h, w) != (224, 224):
            pil_img = Image.fromarray(np.uint8(np.clip(gray_img, 0.0, 1.0) * 255.0)).resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        else:
            pil_img = Image.fromarray(np.uint8(np.clip(gray_img, 0.0, 1.0) * 255.0))

        rgb_pil = pil_img.convert("RGB")
        tensor_x = self.transform(rgb_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.encoder(tensor_x)
            feat = self.pool(feat).flatten()
            return feat.cpu().numpy()


def build_ml_feature_matrix(
    image_paths: List[str],
    mode: str = "foundation_embeddings",
    backbone: str = "resnet50",
    image_size: Tuple[int, int] = (128, 128),
    device: str = "cpu",
) -> np.ndarray:
    """Builds tabular feature matrix for the Classical ML Family across all images.

    Modes:
    - 'radiomic_texture': Handcrafted moments + GLCM + Radial profiles + Gradient edges
    - 'foundation_embeddings': Pretrained deep foundation backbone representations
    - 'hybrid': Concatenation of radiomics and foundation embeddings
    """
    feature_list = []

    if "foundation" in mode or "hybrid" in mode:
        deep_extractor = FoundationFeatureExtractor(backbone_name=backbone, device=device)
    else:
        deep_extractor = None

    for img_path in image_paths:
        res = preprocess_mri_slice(img_path, target_size=image_size)
        gray = res["processed"]

        if mode == "radiomic_texture":
            feats = extract_handcrafted_features_single(gray)
        elif mode == "foundation_embeddings":
            feats = deep_extractor.extract_from_grayscale(gray)
        elif mode == "hybrid":
            f_radio = extract_handcrafted_features_single(gray)
            f_deep = deep_extractor.extract_from_grayscale(gray)
            feats = np.concatenate([f_radio, f_deep])
        else:
            feats = extract_handcrafted_features_single(gray)

        feature_list.append(feats)

    X = np.array(feature_list, dtype=np.float32)
    return X

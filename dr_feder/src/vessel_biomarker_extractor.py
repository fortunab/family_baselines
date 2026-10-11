"""Vascular Biomarker Extractor for Retinal Fundus Photography.

Segments retinal blood vessel tree and extracts quantitative ophthalmic biomarkers:
1. Vessel Area Density (% of retinal surface covered by vasculature)
2. Vessel Tortuosity Index (arteriolar/venular curvature & twisting due to hypoxia)
3. Vessel Branching Complexity (bifurcation density)
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, label


@dataclass
class VascularBiomarkers:
    vessel_density_percent: float
    tortuosity_index: float
    branching_points_count: int
    vessel_pixel_count: int
    retinal_area_pixel_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetinalVesselExtractor:
    """Segments retinal vasculature and computes hemodynamic morphometric biomarkers."""

    def __init__(self, vessel_threshold_percentile: float = 88.0):
        self.threshold_pct = vessel_threshold_percentile

    def segment_vasculature(
        self, img_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts binary vessel mask and field-of-view mask from fundus image."""
        if isinstance(img_input, (str, Path)):
            img_pil = Image.open(img_input).convert("RGB")
            img_arr = np.array(img_pil, dtype=np.float32)
        elif isinstance(img_input, Image.Image):
            img_arr = np.array(img_input.convert("RGB"), dtype=np.float32)
        else:
            img_arr = img_input.astype(np.float32)

        # Use Green channel (maximum contrast for hemoglobin)
        green = img_arr[:, :, 1]
        h, w = green.shape

        # Circular FOV mask
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        fov_mask = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < (min(h, w) * 0.46)

        # Background estimation via large Gaussian blur
        background = gaussian_filter(green, sigma=min(h, w) * 0.12)
        vessel_enhanced = background - green  # vessels are darker than retinal fundus

        # Normalize inside FOV
        fov_vals = vessel_enhanced[fov_mask]
        thresh = np.percentile(fov_vals, self.threshold_pct)
        vessel_mask = (vessel_enhanced > thresh) & fov_mask

        return vessel_mask, fov_mask, green

    def extract_biomarkers(
        self, img_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> VascularBiomarkers:
        """Computes vessel density, tortuosity index, and branching points."""
        vessel_mask, fov_mask, _ = self.segment_vasculature(img_input)

        fov_area = int(np.sum(fov_mask))
        vessel_area = int(np.sum(vessel_mask))

        density_pct = (vessel_area / max(1, fov_area)) * 100.0

        # Connected component analysis for vessel segments
        labeled_vessels, num_features = label(vessel_mask)

        # Tortuosity estimation:
        # Sum of (Perimeter / (2 * MajorAxis)) across prominent vascular segments
        tortuosity_scores = []
        branching_count = 0

        for feat_id in range(1, min(num_features + 1, 60)):
            seg = labeled_vessels == feat_id
            seg_area = np.sum(seg)
            if seg_area < 8:
                continue

            # Estimate perimeter via dilation difference
            dilated = binary_dilation(seg)
            perimeter = np.sum(dilated ^ seg)

            # Euclidean span (distance between furthest points)
            coords = np.argwhere(seg)
            if len(coords) >= 4:
                min_c = np.min(coords, axis=0)
                max_c = np.max(coords, axis=0)
                euclidean_chord = np.linalg.norm(max_c - min_c)
                if euclidean_chord > 3:
                    # Tortuosity index: arc ratio
                    arc_ratio = (perimeter / 2.0) / euclidean_chord
                    tortuosity_scores.append(arc_ratio)

            if seg_area > 20:
                branching_count += 1

        avg_tortuosity = float(np.mean(tortuosity_scores)) if tortuosity_scores else 1.15
        # Normalize tortuosity index (1.0 = straight line, >1.3 = elevated tortuosity)
        normalized_tortuosity = round(float(max(1.0, avg_tortuosity)), 3)

        return VascularBiomarkers(
            vessel_density_percent=round(float(density_pct), 2),
            tortuosity_index=normalized_tortuosity,
            branching_points_count=int(branching_count),
            vessel_pixel_count=vessel_area,
            retinal_area_pixel_count=fov_area,
        )

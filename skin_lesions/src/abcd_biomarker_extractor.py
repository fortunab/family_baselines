"""Quantitative Algorithmic ABCD Rule Biomarker Extractor for Dermoscopy.

Implements the Stolz et al. ABCD Rule of Dermatoscopy:
- A: Asymmetry (Principal axes of inertia, 0 to 2)
- B: Border Irregularity (Octant notch and contour abruptness, 0 to 8)
- C: Color Variegation (Clinical detection of 6 canonical colors, 1 to 6)
- D: Differential Structures / Diameter (Equivalent diameter in mm, 1 to 5)
- TDS: Total Dermoscopy Score = 1.3*A + 0.1*B + 0.5*C + 0.5*D
  - TDS < 4.75: Benign lesion
  - 4.75 <= TDS <= 5.45: Suspicious / Borderline (3-month follow-up)
  - TDS > 5.45: Highly Suspicious / Malignant (Excisional Biopsy indication)
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class ABCDBiomarkers:
    asymmetry_score: int  # 0, 1, or 2
    asymmetry_details: Dict[str, float]
    border_score: int  # 0 to 8
    border_details: Dict[str, Any]
    color_score: int  # 1 to 6
    colors_detected: List[str]
    diameter_mm: float
    diameter_score: int  # 1 to 5
    total_dermoscopy_score: float  # TDS
    clinical_risk_tier: str  # Benign, Suspicious, Malignant Suspect
    lesion_area_fraction: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DermoscopyABCDExtractor:
    """Extracts reproducible quantitative ABCD biomarkers from dermatoscopic images."""

    def __init__(self, pixel_spacing_mm: float = 0.234):
        """pixel_spacing_mm: Calibration factor (default: ~15mm FOV across 64px = 0.234 mm/px)."""
        self.pixel_spacing_mm = pixel_spacing_mm

    def segment_lesion(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segments lesion from background skin using luminance thresholding and morphology."""
        # Convert to luminance
        gray = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
        h, w = gray.shape

        # Avoid edge aperture ring
        y, x = np.ogrid[:h, :w]
        center_dist = np.sqrt((x - w / 2.0) ** 2 + (y - h / 2.0) ** 2)
        valid_fov = center_dist <= (min(h, w) * 0.44)

        fov_pixels = gray[valid_fov]
        thresh = np.percentile(fov_pixels, 42)

        # Lesions are darker than skin
        mask = (gray < thresh) & valid_fov

        # Morphological clean up
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))

        # Keep largest connected component
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = int(np.argmax(sizes) + 1)
            lesion_mask = labeled == largest_label
        else:
            lesion_mask = mask

        return lesion_mask, gray

    def compute_asymmetry(self, mask: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """Calculates Asymmetry (A) along principal axes of inertia.

        Score:
        - 0: Symmetric in both axes
        - 1: Asymmetric in 1 axis
        - 2: Asymmetric in both axes
        """
        area = np.sum(mask)
        if area < 10:
            return 0, {"axis_1_mismatch": 0.0, "axis_2_mismatch": 0.0}

        y_coords, x_coords = np.nonzero(mask)
        cy = np.mean(y_coords)
        cx = np.mean(x_coords)

        # Central moments
        mu20 = np.mean((x_coords - cx) ** 2)
        mu02 = np.mean((y_coords - cy) ** 2)
        mu11 = np.mean((x_coords - cx) * (y_coords - cy))

        # Principal orientation angle
        angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # Rotate mask to align with principal axes
        angle_deg = np.degrees(angle)
        rot_mask = ndimage.rotate(mask.astype(np.float32), -angle_deg, reshape=False, order=0) > 0.5

        # Re-center
        y_r, x_r = np.nonzero(rot_mask)
        if len(y_r) == 0:
            return 0, {"axis_1_mismatch": 0.0, "axis_2_mismatch": 0.0}

        # Flip horizontally (along vertical axis)
        flip_h = np.fliplr(rot_mask)
        diff_h = np.sum(np.logical_xor(rot_mask, flip_h))
        mismatch_h = float(diff_h / (2.0 * area))

        # Flip vertically (along horizontal axis)
        flip_v = np.flipud(rot_mask)
        diff_v = np.sum(np.logical_xor(rot_mask, flip_v))
        mismatch_v = float(diff_v / (2.0 * area))

        score = 0
        if mismatch_h > 0.22:
            score += 1
        if mismatch_v > 0.22:
            score += 1

        return score, {"axis_1_mismatch": round(mismatch_h, 4), "axis_2_mismatch": round(mismatch_v, 4)}

    def compute_border_irregularity(self, mask: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Calculates Border Irregularity (B) across 8 radial octants (0 to 8)."""
        area = np.sum(mask)
        if area < 10:
            return 0, {"octant_irregularities": [0] * 8, "compactness": 1.0}

        y_coords, x_coords = np.nonzero(mask)
        cy = np.mean(y_coords)
        cx = np.mean(x_coords)

        # Calculate radial distances to contour points
        dilated = ndimage.binary_dilation(mask)
        contour = dilated & ~mask
        cy_pts, cx_pts = np.nonzero(contour)

        if len(cy_pts) == 0:
            return 0, {"octant_irregularities": [0] * 8, "compactness": 1.0}

        angles = np.arctan2(cy_pts - cy, cx_pts - cx)
        angles = np.mod(angles, 2 * np.pi)
        radii = np.sqrt((cx_pts - cx) ** 2 + (cy_pts - cy) ** 2)

        octant_bounds = np.linspace(0, 2 * np.pi, 9)
        octant_scores = []

        for i in range(8):
            in_octant = (angles >= octant_bounds[i]) & (angles < octant_bounds[i + 1])
            if np.sum(in_octant) > 2:
                octant_r = radii[in_octant]
                # High radial variance indicates notched, abrupt border
                r_var = np.std(octant_r) / (np.mean(octant_r) + 1e-5)
                is_irregular = int(r_var > 0.18)
            else:
                is_irregular = 0
            octant_scores.append(is_irregular)

        border_score = int(np.sum(octant_scores))
        perimeter = float(len(cy_pts))
        compactness = float((perimeter**2) / (4.0 * np.pi * area))

        return border_score, {
            "octant_irregularities": octant_scores,
            "compactness": round(compactness, 4),
        }

    def compute_color_variegation(self, img_rgb: np.ndarray, mask: np.ndarray) -> Tuple[int, List[str]]:
        """Identifies clinical dermoscopic colors (C, 1 to 6).

        Clinical Colors:
        1. Light brown (tan)
        2. Dark brown
        3. Black
        4. Red (vascularity)
        5. White (regression/fibrosis)
        6. Slate blue / gray (melanophages)
        """
        lesion_pixels = img_rgb[mask]
        n_pixels = len(lesion_pixels)
        if n_pixels < 10:
            return 1, ["light_brown"]

        r = lesion_pixels[:, 0]
        g = lesion_pixels[:, 1]
        b = lesion_pixels[:, 2]

        color_flags = {
            "light_brown": False,
            "dark_brown": False,
            "black": False,
            "red": False,
            "white": False,
            "slate_blue_gray": False,
        }

        # Threshold rules
        # 1. Black (hypopigmentation/dark melanin): r, g, b all low
        black_mask = (r < 65) & (g < 65) & (b < 65)
        if np.sum(black_mask) / n_pixels > 0.03:
            color_flags["black"] = True

        # 2. Dark brown
        db_mask = (r >= 65) & (r < 130) & (g < 95) & (b < 75) & (r > g) & (g > b)
        if np.sum(db_mask) / n_pixels > 0.05:
            color_flags["dark_brown"] = True

        # 3. Light brown (tan)
        lb_mask = (r >= 130) & (r < 205) & (g >= 95) & (g < 165) & (b < 135) & (r > g)
        if np.sum(lb_mask) / n_pixels > 0.05:
            color_flags["light_brown"] = True

        # 4. Red (erythema / vascular lacunae)
        red_mask = (r > 150) & (g < 110) & (b < 110) & ((r - g) > 40)
        if np.sum(red_mask) / n_pixels > 0.02:
            color_flags["red"] = True

        # 5. White (scar-like depigmentation)
        white_mask = (r > 195) & (g > 185) & (b > 180) & (np.abs(r - g) < 25)
        if np.sum(white_mask) / n_pixels > 0.02:
            color_flags["white"] = True

        # 6. Slate Blue / Gray (dermal melanin)
        blue_mask = (b >= (g - 15)) & (b >= (r - 25)) & (r < 140) & (b > 70)
        if np.sum(blue_mask) / n_pixels > 0.02:
            color_flags["slate_blue_gray"] = True

        detected = [k for k, v in color_flags.items() if v]
        score = max(1, min(6, len(detected)))
        return score, detected

    def compute_diameter(self, mask: np.ndarray) -> Tuple[float, int]:
        """Calculates Diameter (D) in mm and clinical score (1 to 5).

        In clinical dermoscopy, lesions > 6mm receive the highest score (5).
        """
        area = np.sum(mask)
        if area < 5:
            return 2.0, 1

        equiv_diameter_px = 2.0 * np.sqrt(area / np.pi)
        diameter_mm = float(equiv_diameter_px * self.pixel_spacing_mm)

        if diameter_mm >= 6.0:
            score = 5
        elif diameter_mm >= 5.0:
            score = 4
        elif diameter_mm >= 4.0:
            score = 3
        elif diameter_mm >= 3.0:
            score = 2
        else:
            score = 1

        return round(diameter_mm, 2), score

    def extract_biomarkers(self, img_input: Union[Image.Image, np.ndarray]) -> ABCDBiomarkers:
        """Extracts complete Stolz ABCD profile and computes Total Dermoscopy Score (TDS)."""
        if isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert("RGB"), dtype=np.float32)
        else:
            img_rgb = img_input.astype(np.float32)

        h, w = img_rgb.shape[:2]
        mask, _ = self.segment_lesion(img_rgb)
        area_fraction = float(np.sum(mask) / (h * w))

        # A: Asymmetry
        score_a, details_a = self.compute_asymmetry(mask)

        # B: Border
        score_b, details_b = self.compute_border_irregularity(mask)

        # C: Color
        score_c, colors_c = self.compute_color_variegation(img_rgb, mask)

        # D: Diameter
        diam_mm, score_d = self.compute_diameter(mask)

        # Total Dermoscopy Score (TDS) = 1.3*A + 0.1*B + 0.5*C + 0.5*D
        tds = 1.3 * score_a + 0.1 * score_b + 0.5 * score_c + 0.5 * score_d
        tds = round(float(tds), 2)

        # Risk Stratification
        if tds < 4.75:
            tier = "Benign Lesion"
        elif tds <= 5.45:
            tier = "Suspicious / Borderline"
        else:
            tier = "Malignant Suspect (Melanoma Indication)"

        return ABCDBiomarkers(
            asymmetry_score=score_a,
            asymmetry_details=details_a,
            border_score=score_b,
            border_details=details_b,
            color_score=score_c,
            colors_detected=colors_c,
            diameter_mm=diam_mm,
            diameter_score=score_d,
            total_dermoscopy_score=tds,
            clinical_risk_tier=tier,
            lesion_area_fraction=round(area_fraction, 4),
        )

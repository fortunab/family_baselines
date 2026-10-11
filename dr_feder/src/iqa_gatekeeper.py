"""Image Quality Assurance (IQA Gatekeeper) for Retinal Fundus Photography.

Screening gatekeeper evaluating whether a retinal fundus scan possesses sufficient
optical clarity, contrast, and focus to be diagnostically gradable for Diabetic Retinopathy.
Prevents false-negative errors caused by cataracts, pupil underexpansion, or motion blur.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image
from scipy.ndimage import laplace


@dataclass
class IQAResult:
    image_path: str
    is_gradable: bool
    quality_score: float  # 0 to 100
    quality_tier: str     # "Good", "Borderline", "Ungradable / Reject"
    sharpness_index: float
    illumination_score: float
    contrast_ratio: float
    detected_defects: List[str]
    clinical_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FundusIQAGatekeeper:
    """Evaluates optical gradability of retinal fundus photographs."""

    def __init__(
        self,
        min_sharpness_threshold: float = 12.0,
        min_contrast_threshold: float = 20.0,
        illumination_bounds: tuple = (40.0, 220.0),
    ):
        self.min_sharpness = min_sharpness_threshold
        self.min_contrast = min_contrast_threshold
        self.illum_bounds = illumination_bounds

    def evaluate_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> IQAResult:
        """Evaluates a fundus photograph and returns a structured clinical IQA report."""
        path_str = ""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            img_pil = Image.open(image_input).convert("RGB")
            img_arr = np.array(img_pil, dtype=np.float32)
        elif isinstance(image_input, Image.Image):
            img_arr = np.array(image_input.convert("RGB"), dtype=np.float32)
        else:
            img_arr = image_input.astype(np.float32)

        # Separate green channel (highest diagnostic contrast for retinal vasculature)
        green_ch = img_arr[:, :, 1]
        gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]

        # 1. Circular Field-of-View Mask (exclude black camera border)
        h, w = gray.shape
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        fov_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        fov_mask = fov_dist < (min(h, w) * 0.46)

        if not np.any(fov_mask):
            fov_mask = np.ones_like(gray, dtype=bool)

        fov_pixels = gray[fov_mask]

        # 2. Illumination Adequacy
        mean_lum = float(np.mean(fov_pixels))
        illum_score = max(0.0, min(100.0, 100.0 - abs(mean_lum - 128.0) * 0.8))

        # 3. Contrast Ratio
        contrast_val = float(np.std(fov_pixels))

        # 4. Sharpness / Focus Index (Laplacian Variance inside FOV)
        lap_resp = laplace(green_ch)
        sharpness_val = float(np.var(lap_resp[fov_mask]))

        # Detect defects
        defects = []
        if sharpness_val < self.min_sharpness:
            defects.append("Motion Blur / Out of Focus")
        if mean_lum < self.illum_bounds[0]:
            defects.append("Severe Underexposure / Small Pupil")
        elif mean_lum > self.illum_bounds[1]:
            defects.append("Overexposure / Corneal Glare")
        if contrast_val < self.min_contrast:
            defects.append("Low Retinal Contrast / Cataract Opacity")

        # Composite Quality Score (0 to 100)
        norm_sharp = min(100.0, (sharpness_val / 30.0) * 100.0)
        norm_contrast = min(100.0, (contrast_val / 40.0) * 100.0)
        composite_score = 0.40 * norm_sharp + 0.35 * norm_contrast + 0.25 * illum_score
        composite_score = round(float(np.clip(composite_score, 0.0, 100.0)), 2)

        is_gradable = len(defects) == 0 and composite_score >= 45.0

        if composite_score >= 70.0:
            tier = "Good"
            rec = "Image quality is optimal; proceed to automated DR staging."
        elif is_gradable:
            tier = "Borderline / Acceptable"
            rec = "Quality acceptable for screening, subtle microaneurysms may have reduced visibility."
        else:
            tier = "Ungradable / Reject"
            rec = "Reject scan: Repeat fundus photography; recommend pharmacologic mydriasis if pupil is small."

        return IQAResult(
            image_path=path_str,
            is_gradable=is_gradable,
            quality_score=composite_score,
            quality_tier=tier,
            sharpness_index=round(sharpness_val, 2),
            illumination_score=round(illum_score, 2),
            contrast_ratio=round(contrast_val, 2),
            detected_defects=defects,
            clinical_recommendation=rec,
        )

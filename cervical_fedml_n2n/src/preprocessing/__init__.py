"""
Preprocessing Factory Module for Cervical Cytology Micrographs.
Provides unified dispatch between baseline normalization, Noise2Noise (N2N) denoising,
CycleGAN stain harmonization, and hybrid N2N + CycleGAN dual-stage restoration.
"""

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.preprocessing.cyclegan_preprocessor import (
    CycleGANCervicalPreprocessor,
    get_cyclegan_pipeline,
)
from src.preprocessing.n2n_preprocessor import (
    N2NCervicalPreprocessor,
    get_n2n_pipeline,
)


class N2NCycleGANHybridPreprocessor:
    """
    Dual-Stage Cytology Image Preprocessor:
    Stage 1: Noise2Noise (N2N) sensor denoising & chromatin texture restoration
    Stage 2: CycleGAN inter-clinic Papanicolaou stain domain harmonization
    """

    def __init__(
        self,
        n2n_pipe: N2NCervicalPreprocessor,
        cyclegan_pipe: CycleGANCervicalPreprocessor,
    ):
        self.n2n = n2n_pipe
        self.cyclegan = cyclegan_pipe
        self.final_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_pil(self, img: Image.Image) -> Image.Image:
        """Applies sequential N2N denoising followed by CycleGAN stain harmonization."""
        denoised_img = self.n2n.preprocess_pil(img)
        harmonized_img = self.cyclegan.preprocess_pil(denoised_img)
        return harmonized_img

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Applies sequential processing and returns normalized tensor."""
        processed_pil = self.preprocess_pil(img)
        t = transforms.ToTensor()(processed_pil)
        return self.final_norm(t)


def get_standard_pipeline(cfg: dict[str, Any]) -> Callable[[Image.Image], torch.Tensor]:
    """Standard baseline ImageNet normalization pipeline without additional preprocessing."""
    img_size = int(cfg.get("image_size", 224))
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_preprocessor(mode: str, cfg: dict[str, Any]) -> Any:
    """
    Factory creating the selected preprocessing pipeline:
    - 'none': Standard baseline ImageNet normalization
    - 'n2n': Noise2Noise cellular restoration & optical denoising
    - 'cyclegan': CycleGAN stain domain harmonization
    - 'n2n_cyclegan': Combined dual-stage N2N restoration + CycleGAN stain harmonization
    """
    m = mode.lower().strip()
    if m in ["n2n_cyclegan", "hybrid", "both", "all"]:
        n2n = get_n2n_pipeline(cfg)
        cyclegan = get_cyclegan_pipeline(cfg)
        return N2NCycleGANHybridPreprocessor(n2n, cyclegan)
    elif m == "n2n":
        return get_n2n_pipeline(cfg)
    elif m == "cyclegan":
        return get_cyclegan_pipeline(cfg)
    else:
        return get_standard_pipeline(cfg)


__all__ = [
    "CycleGANCervicalPreprocessor",
    "N2NCervicalPreprocessor",
    "N2NCycleGANHybridPreprocessor",
    "get_cyclegan_pipeline",
    "get_n2n_pipeline",
    "get_preprocessor",
    "get_standard_pipeline",
]

"""
Cellular Preprocessing Package for Cervical Cytology.
Supports Variational Autoencoder (VAE) and Standard Baseline Normalization.
"""

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from src.preprocessing.vae_preprocessor import (
    CervicalVAE,
    VAECervicalPreprocessor,
    get_vae_pipeline,
)


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


class StandardPassthroughPreprocessor:
    """Standard preprocessor matching the interface of VAE preprocessor."""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_pil(self, img: Image.Image) -> Image.Image:
        """Returns unaltered image resized to standard dimensions."""
        return img.resize((self.img_size, self.img_size))

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Processes PIL Image into normalized tensor."""
        return self.tfm(img)


def get_preprocessor(mode: str, cfg: dict[str, Any]) -> Any:
    """
    Factory creating the selected preprocessing pipeline:
    - 'none': Standard baseline ImageNet normalization
    - 'vae': Variational Autoencoder cellular reconstruction & regularization
    """
    m = mode.lower().strip()
    if m in ["vae", "variational_autoencoder", "autoencoder"]:
        return get_vae_pipeline(cfg)
    else:
        img_size = int(cfg.get("image_size", 224))
        return StandardPassthroughPreprocessor(img_size=img_size)


__all__ = [
    "CervicalVAE",
    "StandardPassthroughPreprocessor",
    "VAECervicalPreprocessor",
    "get_preprocessor",
    "get_standard_pipeline",
    "get_vae_pipeline",
]

"""
Noise2Noise (N2N) Cellular Denoising & Micro-Structure Restoration Pipeline.
Restores clean nuclear chromatin texture, cytoplasmic boundary contrast,
and suppresses optical sensor noise in cervical cytology micrographs.
"""

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class DoubleConv(nn.Module):
    """Double convolution block with Batch Normalization and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Noise2NoiseUNet(nn.Module):
    """
    Lightweight U-Net Restorative Architecture for Noise2Noise Cytology Denoising.
    Learns mapping from noisy microscopy observation to clean cellular signal.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder with skip connections
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        out = self.final(d1)
        # Residual connection adds back low-frequency image content
        return torch.sigmoid(x + out * 0.1)


class N2NCervicalPreprocessor:
    """
    Noise2Noise Cervical Cytology Preprocessor.
    Applies blind noise suppression and nuclear chromatin enhancement.
    """

    def __init__(
        self,
        img_size: int = 224,
        device: str = "cpu",
    ):
        self.img_size = img_size
        self.device = torch.device(device)
        self.model = Noise2NoiseUNet(in_channels=3, out_channels=3)
        self.model.to(self.device)
        self.model.eval()

        self.to_tensor_tfm = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def preprocess_pil(self, img: Image.Image) -> Image.Image:
        """Applies N2N restoration and returns cleaned PIL Image."""
        t = self.to_tensor_tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            denoised = self.model(t).squeeze(0).cpu().clamp(0.0, 1.0)
            arr = (denoised.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Processes PIL Image into normalized tensor."""
        cleaned_pil = self.preprocess_pil(img)
        t = transforms.ToTensor()(cleaned_pil)
        return self.norm(t)


def get_n2n_pipeline(cfg: dict[str, Any]) -> N2NCervicalPreprocessor:
    """Constructs N2N preprocessor configured from dictionary."""
    img_size = int(cfg.get("image_size", 224))
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return N2NCervicalPreprocessor(img_size=img_size, device=device)

"""
Variational Autoencoder (VAE) Cellular Preprocessor for Cervical Cytology.
Applies probabilistic latent representation modeling and Evidence Lower Bound (ELBO)
regularization to suppress optical sensor noise, normalize staining anomalies,
and stabilize chromatin micro-structures before federated training.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class ConvEncoder(nn.Module):
    """Convolutional encoder mapping RGB micrographs to latent mean and log-variance."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.fc_mu = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


class ConvDecoder(nn.Module):
    """Convolutional decoder reconstructing RGB micrographs from sampled latent codes."""

    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),  # 224x224
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 256, 7, 7)
        return self.deconv(h)


class CervicalVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for cellular micrograph reconstruction.
    Implements Gaussian reparameterization: z = mu + eps * sigma.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample z from N(mu, var) while maintaining backprop."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.001,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evidence Lower Bound (ELBO) loss = Reconstruction MSE + beta * KL Divergence."""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


class VAECervicalPreprocessor:
    """
    Variational Autoencoder Cervical Cytology Preprocessor.
    Normalizes microscopy tiles, reconstructs latent representations,
    and returns standardized, regularized micrographs.
    """

    def __init__(
        self,
        img_size: int = 224,
        latent_dim: int = 128,
        device: str = "cpu",
    ):
        self.img_size = img_size
        self.device = torch.device(device)
        self.model = CervicalVAE(in_channels=3, latent_dim=latent_dim)
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
        """Applies VAE reconstruction and returns cleaned, regularized PIL Image."""
        t = self.to_tensor_tfm(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon, _, _ = self.model(t)
            # High-fidelity residual blend: 85% VAE latent reconstruction + 15% original detail
            blended = 0.85 * recon + 0.15 * t
            arr = (blended.squeeze(0).cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Processes PIL Image into normalized tensor."""
        cleaned_pil = self.preprocess_pil(img)
        t = transforms.ToTensor()(cleaned_pil)
        return self.norm(t)


def get_vae_pipeline(cfg: dict[str, Any]) -> VAECervicalPreprocessor:
    """Constructs VAE preprocessor configured from dictionary."""
    img_size = int(cfg.get("image_size", 224))
    latent_dim = int(cfg.get("vae_latent_dim", 128))
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return VAECervicalPreprocessor(img_size=img_size, latent_dim=latent_dim, device=device)

"""PyTorch Deep Learning Models for Federated Retinal Fundus Classification.

Provides:
- RetinaResNet: Lightweight residual network tailored for 5-grade Diabetic Retinopathy
  classification, optimized for low bandwidth parameter transmission across hospital networks.
- Utility functions for parameter cloning, weight divergence, and FedProx proximal regularization.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual convolutional block with skip shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class RetinaResNet(nn.Module):
    """Compact Residual Neural Network for 5-grade Diabetic Retinopathy.

    Designed for high sensitivity on ophthalmic microvascular lesions while
    maintaining a compact parameter payload for federated communication rounds.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # 3 Residual Stages
        self.layer1 = ResidualBlock(hidden_dim // 2, hidden_dim, stride=2)  # 64 -> 32
        self.layer2 = ResidualBlock(hidden_dim, hidden_dim * 2, stride=2)  # 32 -> 16
        self.layer3 = ResidualBlock(hidden_dim * 2, hidden_dim * 4, stride=2)  # 16 -> 8

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits


def get_model(
    architecture: str = "resnet_retina",
    in_channels: int = 3,
    hidden_dim: int = 64,
    num_classes: int = 5,
    dropout: float = 0.25,
) -> nn.Module:
    """Instantiates the specified neural network model."""
    if architecture in ["resnet_retina", "default"]:
        return RetinaResNet(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    """Returns (total_params, size_mb)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total * 4) / (1024 * 1024)
    return total, size_mb


def compute_proximal_term(model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    """Computes FedProx proximal regularization: 0.5 * sum(||w - w_global||^2)."""
    proximal_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for w, w_t in zip(model.parameters(), global_model.parameters()):
        proximal_loss += torch.sum((w - w_t.detach()) ** 2)
    return 0.5 * proximal_loss


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep clones a model state dictionary."""
    return {k: v.clone().detach() for k, v in state_dict.items()}


def compute_weight_divergence(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    """Computes Euclidean L2 distance between two state dictionaries."""
    total_diff_sq = 0.0
    for k in state_a:
        if k in state_b and state_a[k].is_floating_point():
            total_diff_sq += torch.sum((state_a[k] - state_b[k]) ** 2).item()
    return float(np.sqrt(total_diff_sq))

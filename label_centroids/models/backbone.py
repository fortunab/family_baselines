import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetBackbone(nn.Module):
    """
    Two-block convolutional neural network backbone with global average pooling
    as described in Sections 3.2 and 3.3 of the paper.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Convolutional Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Convolutional Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = base_channels * 2  # 64 features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        features = torch.flatten(x, 1)
        return features

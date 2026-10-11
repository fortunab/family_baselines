import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvNetBackbone


class BaselineCNN(nn.Module):
    """
    Standard CNN classifier baseline (Section 3.2).
    Two convolutional blocks, max-pooling, global average pooling,
    and a linear classification layer producing class logits.
    """
    def __init__(self, num_classes: int = 8, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.backbone = ConvNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.classifier = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the softmax classifier head for t-SNE analysis."""
        return self.backbone(x)

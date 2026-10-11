import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Baseline(nn.Module):
    """
    ResNet-50 baseline model adapted for medical image classification (Section 3.2).
    Includes Dropout (0.5) to prevent overfitting in FL setup.
    """
    def __init__(self, num_classes=3, in_channels=3, dropout_rate=0.5):
        super(ResNet50Baseline, self).__init__()
        # Load backbone without default linear classifier
        weights = ResNet50_Weights.DEFAULT
        self.backbone = resnet50(weights=weights)
        
        # Adapt input conv if in_channels != 3
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

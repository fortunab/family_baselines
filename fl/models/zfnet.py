import torch
import torch.nn as nn

class ZFNetBaseline(nn.Module):
    """
    ZFNet (Zeiler & Fergus) architecture benchmark model (Section 3.2).
    Features 7x7 conv1 with stride 2 and max pooling.
    """
    def __init__(self, num_classes=3, in_channels=3, dropout_rate=0.5):
        super(ZFNetBaseline, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 7x7 conv, stride 2 (Zeiler-Fergus modification over AlexNet 11x11 stride 4)
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

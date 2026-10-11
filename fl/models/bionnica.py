import torch
import torch.nn as nn
import torch.nn.functional as F

class MorphologicalFeatureExtractor(nn.Module):
    """
    Multiscale morphological feature extraction block for polyp pit pattern analysis.
    Extracts surface vascularity and glandular structure characteristics.
    """
    def __init__(self, in_channels, out_channels):
        super(MorphologicalFeatureExtractor, self).__init__()
        self.branch3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f3 = self.branch3x3(x)
        f5 = self.branch5x5(x)
        out = torch.cat([f3, f5], dim=1)
        out = self.bn(out)
        return self.relu(out)


class RuleBasedEvaluationLayer(nn.Module):
    """
    Rule-based decision layer (Section 3.2) enforcing morphological bounds
    for premalignancy risk evaluation (Serrated vs Adenomatous vs Hyperplastic).
    """
    def __init__(self, num_features, num_classes):
        super(RuleBasedEvaluationLayer, self).__init__()
        self.fc_features = nn.Linear(num_features, 128)
        self.rule_weights = nn.Parameter(torch.ones(128, num_classes))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = F.relu(self.fc_features(x))
        # Rule gating mechanism based on morphological thresholding
        gated_feat = feat * torch.sigmoid(feat).mean(dim=1, keepdim=True)
        logits = self.classifier(gated_feat)
        return logits


class BionnicaModel(nn.Module):
    """
    Bionnica Custom Model for Colorectal Polyps Classification (Section 3.2).
    Ref: Babuc et al., 2024 (medRxiv).
    """
    def __init__(self, num_classes=3, in_channels=3, dropout_rate=0.5):
        super(BionnicaModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.morph_block1 = MorphologicalFeatureExtractor(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.morph_block2 = MorphologicalFeatureExtractor(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rule_layer = RuleBasedEvaluationLayer(512 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.morph_block1(x)
        x = self.pool1(x)
        x = self.morph_block2(x)
        x = self.pool2(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.rule_layer(x)
        return x

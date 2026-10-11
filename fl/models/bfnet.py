import torch
import torch.nn as nn
import torch.nn.functional as F

class CytologyFeatureExtractor(nn.Module):
    """
    Cytological feature extraction block detecting nuclear-cytoplasmic ratio
    and hyperchromasia intensity variations in cervical pap smear cells.
    """
    def __init__(self, in_channels, out_channels):
        super(CytologyFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f7 = self.conv7x7(x)
        out = torch.cat([f1, f3, f7], dim=1)
        return self.relu(self.bn(out))


class CytologyRuleDecisionLayer(nn.Module):
    """
    Rule-based classification layer (Section 3.2) evaluating cell abnormality
    indices (dyskeratotic, koilocytic, metaplastic, parabasal, columnar).
    """
    def __init__(self, num_features, num_classes):
        super(CytologyRuleDecisionLayer, self).__init__()
        self.dense = nn.Linear(num_features, 256)
        self.rule_gate = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        h = F.relu(self.dense(x))
        gate = torch.sigmoid(self.rule_gate(h))
        gated_h = h * gate
        logits = self.classifier(gated_h)
        return logits


class BFNetModel(nn.Module):
    """
    BFNet Custom Model for Cervical Cytology Image Classification (Section 3.2).
    Ref: Babuc & Fortiş, 2024 (Springer).
    """
    def __init__(self, num_classes=5, in_channels=3, dropout_rate=0.5):
        super(BFNetModel, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.block1 = CytologyFeatureExtractor(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.block2 = CytologyFeatureExtractor(128, 256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.block3 = CytologyFeatureExtractor(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.rule_layer = CytologyRuleDecisionLayer(512 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.rule_layer(x)
        return x

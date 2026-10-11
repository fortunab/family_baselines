"""Multimodal Fusion Classifier: Dermoscopy Image + Patient Clinical Metadata.

Fuses high-resolution visual convolutional embeddings (ResNet) with tabular clinical metadata:
- Patient Age (continuous)
- Biological Sex (categorical)
- Anatomical Lesion Localization (categorical: back, lower extremity, face, etc.)
- Fitzpatrick Skin Phototype (ordinal: Types I to VI)

Demonstrates the clinical diagnostic synergy between visual morphology and clinical context.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models import SkinLesionResNet

LOCALIZATIONS = [
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "neck",
]


class ClinicalMetadataEncoder(nn.Module):
    """Encodes tabular patient metadata into a latent clinical embedding vector."""

    def __init__(self, output_dim: int = 32):
        super().__init__()
        # Input dimension: age (1) + sex (1) + localization (8) + fitzpatrick (1) = 11
        input_dim = 1 + 1 + len(LOCALIZATIONS) + 1

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, meta_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(meta_tensor)


class MultimodalDermoscopyNet(nn.Module):
    """End-to-End Multimodal Fusion Architecture for Dermoscopy."""

    def __init__(
        self,
        num_classes: int = 7,
        hidden_dim: int = 64,
        meta_embedding_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.visual_backbone = SkinLesionResNet(
            in_channels=3,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        visual_feat_dim = hidden_dim * 4  # 256

        self.metadata_encoder = ClinicalMetadataEncoder(output_dim=meta_embedding_dim)

        # Fusion classification head
        fused_dim = visual_feat_dim + meta_embedding_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 256-dim visual feature representation from SkinLesionResNet."""
        out = self.visual_backbone.stem(x)
        out = self.visual_backbone.layer1(out)
        out = self.visual_backbone.layer2(out)
        out = self.visual_backbone.layer3(out)
        out = self.visual_backbone.pool(out)
        features = torch.flatten(out, 1)
        return features

    def forward(
        self,
        image_tensor: torch.Tensor,
        metadata_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass supporting both multimodal and image-only execution."""
        visual_feats = self.extract_visual_features(image_tensor)

        if metadata_tensor is None:
            # Fall back to image-only linear classification via base fc
            return self.visual_backbone.fc(self.visual_backbone.dropout(visual_feats))

        meta_feats = self.metadata_encoder(metadata_tensor)
        fused = torch.cat([visual_feats, meta_feats], dim=1)
        logits = self.fusion_classifier(fused)
        return logits


def encode_patient_metadata(
    age: int,
    sex: str,
    localization: str,
    fitzpatrick_tone: str = "Type_II",
    localizations_list: Optional[List[str]] = None,
) -> np.ndarray:
    """Converts patient attributes into an 11-dimensional normalized numerical vector."""
    if localizations_list is None:
        localizations_list = LOCALIZATIONS

    # 1. Normalized age [0, 1]
    norm_age = float(np.clip(age / 100.0, 0.0, 1.0))

    # 2. Sex binary (female = 1.0, male = 0.0)
    sex_val = 1.0 if str(sex).lower().startswith("f") else 0.0

    # 3. One-hot localization (len=8)
    loc_vec = [0.0] * len(localizations_list)
    loc_str = str(localization).lower().strip()
    if loc_str in localizations_list:
        loc_vec[localizations_list.index(loc_str)] = 1.0

    # 4. Fitzpatrick skin tone [0, 1]
    fitz_map = {
        "Type_I": 0.166,
        "Type_II": 0.333,
        "Type_III": 0.500,
        "Type_IV": 0.666,
        "Type_V": 0.833,
        "Type_VI": 1.000,
    }
    tone_val = fitz_map.get(fitzpatrick_tone, 0.333)

    vector = [norm_age, sex_val] + loc_vec + [tone_val]
    return np.array(vector, dtype=np.float32)

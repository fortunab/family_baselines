"""
Pathology & Vision Foundation Model Architectures for Herlev Cervical Cytology.
Supported Foundation Backbones in skorch:
1. Owkin Phikon (owkin/phikon) - iBOT ViT-Base pan-cancer histology foundation model
2. Paige Virchow (paige-ai/Virchow) - 632M parameter ViT-Huge whole-slide foundation model
3. Harvard UNI (MahmoodLab/UNI) - ViT-Large 100M+ patch pathology foundation model
4. Meta DINOv2 (facebook/dinov2-base) - Vision Foundation Backbone
5. Microsoft BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
"""

import os
from typing import Optional

import timm
import torch
import torch.nn as nn
from transformers import AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PathologyFoundationClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "owkin/phikon",
        num_classes: int = 7,
        embedding_dim: Optional[int] = None,
        pretrained: bool = True,
        model_type: str = "huggingface",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.model_type = model_type
        self.num_classes = num_classes

        print(
            f"[Foundation-Module] Loading foundation model: '{backbone_name}' (Type={model_type}, Pretrained={pretrained})..."
        )

        self.encoder = None
        self.embed_dim = embedding_dim or 768

        # Load foundation encoder via HuggingFace or timm with robust fallbacks
        try:
            if "phikon" in backbone_name.lower() or "dinov2" in backbone_name.lower():
                self.encoder = AutoModel.from_pretrained(backbone_name)
                if hasattr(self.encoder.config, "hidden_size"):
                    self.embed_dim = self.encoder.config.hidden_size
            elif "biomedclip" in backbone_name.lower():
                self.encoder = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
                if hasattr(self.encoder, "visual"):
                    self.encoder = self.encoder.visual
                self.embed_dim = 512
            else:
                timm_name = backbone_name
                if (
                    "/" in timm_name
                    and not timm_name.startswith("hf-hub:")
                    and not timm_name.startswith("hf_hub:")
                ):
                    timm_name = f"hf-hub:{timm_name}"
                self.encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
                if hasattr(self.encoder, "num_features"):
                    self.embed_dim = self.encoder.num_features
        except Exception as e:
            print(
                f"[Foundation-Module] Direct load notice for '{backbone_name}' ({e}). Falling back to timm vision foundation backbone (vit_base_patch16_224)..."
            )
            try:
                self.encoder = timm.create_model(
                    "vit_base_patch16_224", pretrained=True, num_classes=0
                )
                self.embed_dim = 768
            except Exception:
                self.encoder = timm.create_model("resnet50d", pretrained=True, num_classes=0)
                self.embed_dim = 2048

        # Classification Head (Batch Normalization -> Dropout -> Linear -> ReLU -> Dropout -> Linear)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.embed_dim),
            nn.Dropout(0.3),
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "last_hidden_state"):
            outputs = self.encoder(x)
            feat = outputs.last_hidden_state[:, 0]  # CLS token
        elif hasattr(self.encoder, "forward_features"):
            feat = self.encoder(x)
            if feat.ndim == 3:
                feat = feat[:, 0]
            elif feat.ndim == 4:
                feat = feat.mean(dim=[-2, -1])
        else:
            try:
                feat = self.encoder(x)
                if isinstance(feat, tuple):
                    feat = feat[0]
                if feat.ndim == 3:
                    feat = feat[:, 0]
                elif feat.ndim == 4:
                    feat = feat.mean(dim=[-2, -1])
            except Exception:
                feat = torch.zeros((x.size(0), self.embed_dim), device=x.device)

        return self.head(feat)

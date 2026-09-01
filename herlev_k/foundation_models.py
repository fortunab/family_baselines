"""
Pathology & Vision Foundation Model Factory for Herlev Cervical Cytology.
Supports Owkin Phikon, Paige Virchow, Meta DINOv2, and Torchvision ViT-Large.
"""

import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


class FoundationEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, model_type: str, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.model_type = model_type
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        elif isinstance(out, torch.Tensor):
            return out[:, 0] if out.ndim == 3 else out
        return out


def create_foundation_model(
    model_name: str = "owkin/phikon",
    hf_token: Optional[str] = None
) -> Tuple[FoundationEncoder, Dict[str, Any]]:
    model_config = {
        "model_name": model_name,
        "img_size": 224,
        "embed_dim": 768,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }

    # 1. HuggingFace transformers (Phikon / Virchow / DINOv2)
    try:
        from transformers import AutoModel
        hf_id = "owkin/phikon" if "phikon" in model_name.lower() else model_name
        print(f"[Foundation-Model] Loading '{hf_id}' from HuggingFace Hub...")
        backbone = AutoModel.from_pretrained(hf_id, token=hf_token)
        backbone.eval()
        embed_dim = backbone.config.hidden_size if hasattr(backbone, 'config') and hasattr(backbone.config, 'hidden_size') else 768
        model_config["embed_dim"] = embed_dim
        model_config["model_name"] = hf_id
        return FoundationEncoder(backbone, "hf", embed_dim=embed_dim), model_config
    except Exception as e:
        print(f"[Foundation-Model] HuggingFace load returned ({e}), trying timm/Torchvision...")

    # 2. timm foundation
    try:
        import timm
        timm_name = "vit_base_patch14_dinov2.lvd142m" if "dino" in model_name.lower() else model_name
        print(f"[Foundation-Model] Loading '{timm_name}' via timm...")
        backbone = timm.create_model(timm_name, pretrained=True, num_classes=0)
        backbone.eval()
        embed_dim = backbone.num_features if hasattr(backbone, 'num_features') else 768
        model_config["embed_dim"] = embed_dim
        model_config["model_name"] = timm_name
        return FoundationEncoder(backbone, "timm", embed_dim=embed_dim), model_config
    except Exception as e:
        print(f"[Foundation-Model] timm load returned ({e}), falling back to Torchvision ViT-Large...")

    # 3. Torchvision ViT-Large fallback
    backbone = tv_models.vit_l_16(weights=tv_models.ViT_L_16_Weights.DEFAULT)
    backbone.heads = nn.Identity()
    backbone.eval()
    model_config["embed_dim"] = 1024
    model_config["model_name"] = "torchvision_vit_l_16_foundation"
    return FoundationEncoder(backbone, "torchvision", embed_dim=1024), model_config

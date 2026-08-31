"""
Computational Pathology Foundation Model Factory.
Supports:
1. Virchow & Virchow 2 (Paige AI & Microsoft Research) - 1.5M WSI ViT-Giant (632M params, 1280 dims)
2. Phikon & Phikon-v2 (Owkin / Nature Communications) - TCGA WSI ViT-Base (768 dims)
3. Open-Access Pathology Foundation Fallbacks
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import transforms


class PathologyFoundationEncoder(nn.Module):
    """
    Wraps pre-trained Computational Pathology Foundation Models
    for feature embedding extraction and linear probing.
    """
    def __init__(self, backbone: nn.Module, model_type: str, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.model_type = model_type
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 1D normalized feature representation per image.
        Returns: (B, embed_dim) float tensor.
        """
        if self.model_type in ["virchow", "virchow2"]:
            # Virchow output handling
            output = self.backbone(x)
            if hasattr(output, 'last_hidden_state'):
                # Extract CLS token
                class_token = output.last_hidden_state[:, 0]
            elif isinstance(output, torch.Tensor):
                class_token = output[:, 0] if output.ndim == 3 else output
            else:
                class_token = output
            return class_token

        elif self.model_type in ["phikon", "phikon2"]:
            # Phikon output handling (HuggingFace ViT)
            output = self.backbone(x)
            if hasattr(output, 'last_hidden_state'):
                class_token = output.last_hidden_state[:, 0]
            else:
                class_token = output
            return class_token

        else:
            # Generic ViT / timm backbone
            output = self.backbone(x)
            if output.ndim == 3:
                output = output[:, 0]
            return output


def create_pathology_foundation_model(
    model_name: str = "paige-ai/Virchow",
    hf_token: Optional[str] = None
) -> Tuple[PathologyFoundationEncoder, Dict[str, Any]]:
    """
    Loads pre-trained Computational Pathology Foundation Model.
    Attempts Paige Virchow / Virchow 2, with automatic fallback to Owkin Phikon (non-gated).
    """
    model_config = {
        "model_name": model_name,
        "img_size": 224,
        "embed_dim": 1280 if "virchow" in model_name.lower() else 768,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }

    # 1. Try loading Virchow / Virchow 2 via timm or HuggingFace
    if "virchow" in model_name.lower():
        try:
            import timm
            print(f"[Pathology-Model] Attempting to load Paige '{model_name}' via timm/HuggingFace Hub...")
            # Virchow is registered as hf-hub:paige-ai/Virchow or paige-ai/Virchow2
            hub_name = f"hf-hub:{model_name}" if not model_name.startswith("hf-hub:") else model_name
            backbone = timm.create_model(
                hub_name,
                pretrained=True,
                mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=torch.nn.SiLU,
                hf_token=hf_token
            )
            backbone.eval()
            print(f"[Pathology-Model] Successfully loaded Paige Virchow ViT-Giant (1280-dim)! Pre-trained on 1.5M WSIs.")
            return PathologyFoundationEncoder(backbone, "virchow", embed_dim=1280), model_config

        except Exception as e:
            print(f"[Pathology-Model] Note: Paige Virchow gated access or timm load returned: {e}")
            print("[Pathology-Model] Switching to Owkin Phikon (open-access Pathology Foundation Model pre-trained on TCGA)...")
            model_name = "owkin/phikon"

    # 2. Try loading Owkin Phikon (ViT-Base pre-trained on 40M+ TCGA pathology tiles)
    if "phikon" in model_name.lower():
        try:
            from transformers import AutoModel, AutoImageProcessor
            print(f"[Pathology-Model] Loading Owkin Phikon from HuggingFace Hub ('{model_name}')...")
            backbone = AutoModel.from_pretrained(model_name, token=hf_token)
            backbone.eval()
            model_config["embed_dim"] = 768
            model_config["model_name"] = model_name
            print(f"[Pathology-Model] Successfully loaded Owkin Phikon ViT-Base (768-dim)!")
            return PathologyFoundationEncoder(backbone, "phikon", embed_dim=768), model_config
        except Exception as e:
            print(f"[Pathology-Model] transformers load failed ({e}). Falling back to Torchvision Foundation...")

    # 3. Fallback: DINO / ViT-Large (ImageNet / SSL)
    import torchvision.models as tv_models
    print("[Pathology-Model] Using Vision Foundation ViT-Large backbone as local fallback...")
    backbone = tv_models.vit_l_16(weights=tv_models.ViT_L_16_Weights.DEFAULT)
    # Remove head
    backbone.heads = nn.Identity()
    backbone.eval()
    model_config["embed_dim"] = 1024
    model_config["model_name"] = "vit_large_foundation_fallback"
    return PathologyFoundationEncoder(backbone, "generic", embed_dim=1024), model_config


if __name__ == "__main__":
    encoder, cfg = create_pathology_foundation_model("paige-ai/Virchow")
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        emb = encoder(dummy)
    print(f"[Sanity Test] Pathology Foundation embedding shape: {emb.shape} (Expected: [2, {cfg['embed_dim']}])")

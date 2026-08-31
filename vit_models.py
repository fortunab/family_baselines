"""
Vision Transformer Model Factory.
Supports State-of-the-Art Vision Foundation Architectures:
- EVA-02 (e.g. eva02_base_patch14_448, eva02_tiny_patch14_336)
- Vision Transformers (e.g. vit_base_patch16_224, vit_large_patch14_224)
- Swin Transformers (e.g. swin_base_patch4_window7_224)
- Native Torchvision fallback backbones (vit_b_16, vit_l_16, swin_b)
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


def create_vit_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 8,
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Creates and initializes a Vision Transformer architecture for 8-class histological classification.
    Returns:
    - model: nn.Module with adapted classification head
    - model_config: Dictionary containing native resolution, mean, std, and patch size.
    """
    model_config = {
        "model_name": model_name,
        "num_classes": num_classes,
        "img_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }

    # 1. Try loading via timm (PyTorch Image Models)
    try:
        import timm
        print(f"[ViT-Model] Attempting to load '{model_name}' via timm...")
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        
        # Extract native image size & normalization stats from timm model default_cfg
        default_cfg = getattr(model, 'default_cfg', {})
        if 'input_size' in default_cfg:
            model_config['img_size'] = default_cfg['input_size'][-1]
        if 'mean' in default_cfg:
            model_config['mean'] = default_cfg['mean']
        if 'std' in default_cfg:
            model_config['std'] = default_cfg['std']
            
        print(f"[ViT-Model] Successfully loaded '{model_name}' via timm! Target resolution: {model_config['img_size']}x{model_config['img_size']}")
        return model, model_config

    except (ImportError, Exception) as e:
        print(f"[ViT-Model] timm not available or model not in timm ({e}). Falling back to Torchvision...")

    # 2. Torchvision Fallback Models
    if "large" in model_name.lower() or "vit_l" in model_name.lower():
        print("[ViT-Model] Initializing Torchvision ViT-Large/16 (vit_l_16)...")
        weights = tv_models.ViT_L_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_l_16(weights=weights)
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_vit_l_16"
        model_config["img_size"] = 224

    elif "swin" in model_name.lower():
        print("[ViT-Model] Initializing Torchvision Swin-Base (swin_b)...")
        weights = tv_models.Swin_B_Weights.DEFAULT if pretrained else None
        model = tv_models.swin_b(weights=weights)
        in_feats = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_swin_b"
        model_config["img_size"] = 224

    else:
        # Default standard ViT-Base/16
        print("[ViT-Model] Initializing Torchvision ViT-Base/16 (vit_b_16)...")
        weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_feats = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_vit_b_16"
        model_config["img_size"] = 224

    return model, model_config


def get_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.05
) -> list:
    """
    Creates parameter groups with differential learning rates and decoupled weight decay.
    Classification head gets higher LR, backbone gets lower fine-tuning LR.
    Biases and LayerNorm parameters are excluded from weight decay.
    """
    head_names = ["head", "heads", "fc", "classifier"]
    
    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_head = any(hn in name.lower() for hn in head_names)
        is_no_decay = (len(param.shape) <= 1) or name.endswith(".bias") or ("norm" in name.lower())

        if is_head:
            if is_no_decay:
                no_decay_head.append(param)
            else:
                decay_head.append(param)
        else:
            if is_no_decay:
                no_decay_backbone.append(param)
            else:
                decay_backbone.append(param)

    param_groups = [
        {"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": decay_head, "lr": head_lr, "weight_decay": weight_decay},
        {"params": no_decay_head, "lr": head_lr, "weight_decay": 0.0}
    ]

    # Filter out empty parameter groups
    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    return param_groups


if __name__ == "__main__":
    model, cfg = create_vit_model(model_name="vit_base_patch16_224", num_classes=8)
    dummy_input = torch.randn(2, 3, cfg['img_size'], cfg['img_size'])
    out = model(dummy_input)
    print(f"[Sanity Test] Model output shape: {out.shape} (Expected: [2, 8])")

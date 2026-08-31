"""
ConvNeXt Model Factory (ConvNeXt-Tiny / ConvNeXt-Small / ConvNeXt-Base).
Implements Meta AI's modernized pure Convolutional Network (Liu et al. 2022)
adapted for 8-class colorectal histology classification.
Supports timm backbones and native Torchvision fallbacks.
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


def create_convnext_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 8,
    pretrained: bool = True,
    drop_rate: float = 0.2
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Initializes a ConvNeXt model with an adapted 8-class classification head.
    Returns:
    - model: nn.Module
    - model_config: Dictionary with image resolution, mean, and std.
    """
    model_config = {
        "model_name": model_name,
        "num_classes": num_classes,
        "img_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }

    # 1. Try loading via timm
    try:
        import timm
        print(f"[ConvNeXt-Model] Attempting to load '{model_name}' via timm...")
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        
        default_cfg = getattr(model, 'default_cfg', {})
        if 'input_size' in default_cfg:
            model_config['img_size'] = default_cfg['input_size'][-1]
        if 'mean' in default_cfg:
            model_config['mean'] = default_cfg['mean']
        if 'std' in default_cfg:
            model_config['std'] = default_cfg['std']
            
        print(f"[ConvNeXt-Model] Successfully loaded '{model_name}' via timm! Target resolution: {model_config['img_size']}x{model_config['img_size']}")
        return model, model_config

    except (ImportError, Exception) as e:
        print(f"[ConvNeXt-Model] timm not available or model not in timm ({e}). Falling back to Torchvision...")

    # 2. Torchvision Native Fallback
    if "small" in model_name.lower():
        print("[ConvNeXt-Model] Initializing Torchvision ConvNeXt-Small...")
        weights = tv_models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_small(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_convnext_small"

    elif "base" in model_name.lower():
        print("[ConvNeXt-Model] Initializing Torchvision ConvNeXt-Base...")
        weights = tv_models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_base(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_convnext_base"

    else:
        # Default ConvNeXt-Tiny
        print("[ConvNeXt-Model] Initializing Torchvision ConvNeXt-Tiny...")
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        in_feats = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_feats, num_classes)
        )
        model_config["model_name"] = "torchvision_convnext_tiny"

    return model, model_config


def get_convnext_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 2e-5,
    head_lr: float = 2e-4,
    weight_decay: float = 0.05
) -> list:
    """
    Creates parameter groups with differential learning rates for ConvNeXt.
    Decouples weight decay from biases, LayerNorm, and 1D parameters.
    """
    head_names = ["classifier", "head", "fc"]
    
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

    param_groups = [pg for pg in param_groups if len(pg["params"]) > 0]
    return param_groups


if __name__ == "__main__":
    model, cfg = create_convnext_model(model_name="convnext_tiny", num_classes=8)
    dummy_input = torch.randn(2, 3, cfg['img_size'], cfg['img_size'])
    out = model(dummy_input)
    print(f"[Sanity Test] ConvNeXt output shape: {out.shape} (Expected: [2, 8])")

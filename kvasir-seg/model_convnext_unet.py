"""
Meta ConvNeXt-Base U-Net Semantic Segmentation Foundation Model Pipeline.
Adapted for polyp segmentation on Kvasir-SEG.
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtUNetModel(nn.Module):
    def __init__(self, model_name: str = "convnext_base"):
        super().__init__()
        self.model_name = model_name
        self.encoder = None
        self._init_backbone()

    def _init_backbone(self):
        try:
            import timm
            print(f"[ConvNeXt-UNet] Loading timm pretrained backbone: '{self.model_name}'...")
            self.encoder = timm.create_model(self.model_name, pretrained=True, features_only=True)
            feature_dims = self.encoder.feature_info.channels()  # e.g., [128, 256, 512, 1024]
            print(f"[ConvNeXt-UNet] Feature channels: {feature_dims}")

            # U-Net Style Feature Pyramid Decoder
            self.dec4 = nn.Sequential(nn.Conv2d(feature_dims[3] + feature_dims[2], 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU())
            self.dec3 = nn.Sequential(nn.Conv2d(256 + feature_dims[1], 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
            self.dec2 = nn.Sequential(nn.Conv2d(128 + feature_dims[0], 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.GELU())
            self.final_conv = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 1, kernel_size=1)
            )
        except Exception as e:
            print(f"[ConvNeXt-UNet] Notice ({e}), building custom ConvNeXt block U-Net...")
            self.encoder = None
            # Inverted Bottleneck custom ConvNeXt architecture
            self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=4), nn.BatchNorm2d(64))
            self.stage1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64), nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, 128, kernel_size=1))
            self.stage2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128), nn.BatchNorm2d(128), nn.GELU(), nn.Conv2d(128, 256, kernel_size=1))
            self.stage3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 512, kernel_size=1))
            
            self.u3 = nn.Sequential(nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.GELU())
            self.u2 = nn.Sequential(nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.GELU())
            self.final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.encoder is not None:
            features = self.encoder(x)  # [e0, e1, e2, e3]
            f0, f1, f2, f3 = features[0], features[1], features[2], features[3]

            d3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
            d3 = self.dec4(torch.cat([d3, f2], dim=1))

            d2 = F.interpolate(d3, size=f1.shape[2:], mode='bilinear', align_corners=False)
            d2 = self.dec3(torch.cat([d2, f1], dim=1))

            d1 = F.interpolate(d2, size=f0.shape[2:], mode='bilinear', align_corners=False)
            d1 = self.dec2(torch.cat([d1, f0], dim=1))

            out = F.interpolate(d1, size=(h, w), mode='bilinear', align_corners=False)
            return self.final_conv(out)

        # Fallback forward
        s0 = self.stem(x)       # [B, 64, H/4, W/4]
        s1 = self.stage1(s0)    # [B, 128, H/4, W/4]
        s2 = self.stage2(s1)    # [B, 256, H/8, W/8]
        s3 = self.stage3(s2)    # [B, 512, H/16, W/16]

        u2 = F.interpolate(s3, size=s2.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.u3(torch.cat([u2, s2], dim=1))

        u1 = F.interpolate(u2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        u1 = self.u2(torch.cat([u1, s1], dim=1))

        out = F.interpolate(u1, size=(h, w), mode='bilinear', align_corners=False)
        return self.final(out)

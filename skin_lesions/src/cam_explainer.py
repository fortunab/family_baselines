"""Grad-CAM and Grad-CAM++ Visual Explainability for Skin Lesion Classification.

Extracts pixel-level spatial attention heatmaps from the convolutional residual stages
of SkinLesionResNet, highlighting hallmark dermatoscopic structures:
- Atypical pigment network
- Streaks / pseudopods at lesion margins
- Blue-white veil
- Irregular blotches and vascular lacunae
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAMExplainer:
    """Computes Grad-CAM and Grad-CAM++ attention maps for SkinLesionResNet."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to layer3 (the last residual convolutional stage)
            if hasattr(model, "layer3"):
                self.target_layer = model.layer3
            elif hasattr(model, "visual_backbone") and hasattr(model.visual_backbone, "layer3"):
                self.target_layer = model.visual_backbone.layer3
            else:
                raise ValueError("Could not automatically locate layer3 in model.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        use_gradcam_pp: bool = True,
    ) -> np.ndarray:
        """Generates normalized 2D attention heatmap in range [0, 1]."""
        self.model.zero_grad()
        logits = self.model(image_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        activations = self.activations.detach()  # [1, C, H, W]
        gradients = self.gradients.detach()  # [1, C, H, W]

        if use_gradcam_pp:
            # Grad-CAM++: Pixel-weighted positive gradients
            grads_power_2 = gradients**2
            grads_power_3 = gradients**3
            sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
            eps = 1e-7

            alpha_numer = grads_power_2
            alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3 + eps
            alphas = alpha_numer / alpha_denom

            weights = torch.sum(alphas * F.relu(gradients), dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        min_val = np.min(cam_np)
        max_val = np.max(cam_np)

        if max_val - min_val > 1e-8:
            heatmap = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap

    def overlay_heatmap_on_dermoscopy(
        self,
        img_input: Union[Image.Image, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Interpolates heatmap to original resolution and blends with dermoscopy image."""
        if isinstance(img_input, Image.Image):
            base_img = np.array(img_input.convert("RGB"), dtype=np.float32) / 255.0
        else:
            base_img = img_input.copy().astype(np.float32)
            if np.max(base_img) > 1.0:
                base_img /= 255.0

        h, w = base_img.shape[:2]

        # Resize heatmap with bilinear interpolation via PIL
        heat_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        heat_resized = np.array(heat_pil, dtype=np.float32) / 255.0

        # Pseudo-color colormap (turbo/jet gradient)
        # Low: blue (0,0,1), Mid: green/yellow (0,1,0 -> 1,1,0), High: red (1,0,0)
        cmap = np.zeros((h, w, 3), dtype=np.float32)
        cmap[:, :, 0] = np.clip(1.5 * heat_resized - 0.5, 0.0, 1.0)
        cmap[:, :, 1] = np.clip(1.0 - 2.0 * np.abs(heat_resized - 0.5), 0.0, 1.0)
        cmap[:, :, 2] = np.clip(1.5 * (1.0 - heat_resized) - 0.5, 0.0, 1.0)

        # Alpha blend
        blended = (1.0 - alpha) * base_img + alpha * cmap
        blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
        return blended_uint8

    def close(self):
        """Removes hook handles to prevent memory leaks."""
        self.fwd_handle.remove()
        self.bwd_handle.remove()

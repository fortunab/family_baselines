"""Clinical Explainability Suite for Diabetic Retinopathy (Grad-CAM & Grad-CAM++).

Provides visual spatial explanations indicating which retinal microvascular lesions
(microaneurysms, hard exudates, cotton wool spots, neovascularization) drove the model's DR staging.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExplainer:
    """Computes Grad-CAM and Grad-CAM++ heatmaps for RetinaResNet architectures."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # If target layer is not specified, default to layer3 (penultimate residual stage)
        if target_layer is None:
            if hasattr(model, "layer3"):
                self.target_layer = model.layer3
            else:
                # Find last conv block
                convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                self.target_layer = convs[-1] if convs else model
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

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
        """Generates 2D normalized [0, 1] CAM heatmap for the target severity grade."""
        self.model.zero_grad()

        # Ensure batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)

        logits = self.model(image_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        grads = self.gradients[0]  # (C, H, W)
        acts = self.activations[0]  # (C, H, W)

        if use_gradcam_pp:
            # Grad-CAM++ weights: alpha_k = grads^2 / (2*grads^2 + sum(acts*grads^3))
            grads_power_2 = grads**2
            grads_power_3 = grads_power_2 * grads
            sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)
            eps = 1e-7
            aij = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + eps)
            weights = torch.sum(aij * F.relu(grads), dim=(1, 2))
        else:
            # Standard Grad-CAM: global average pooling of gradients
            weights = torch.mean(grads, dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam_norm = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam)

        return cam_norm

    def overlay_heatmap_on_fundus(
        self,
        original_image: Union[str, Path, Image.Image, np.ndarray],
        cam_heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Overlays resized CAM heatmap on fundus image."""
        if isinstance(original_image, (str, Path)):
            img_pil = Image.open(original_image).convert("RGB")
        elif isinstance(original_image, Image.Image):
            img_pil = original_image.convert("RGB")
        else:
            img_pil = Image.fromarray(np.uint8(original_image))

        w, h = img_pil.size
        # Resize heatmap to match image dimensions
        heatmap_img = Image.fromarray(np.uint8(cam_heatmap * 255)).resize((w, h), Image.Resampling.BICUBIC)
        heatmap_arr = np.array(heatmap_img, dtype=np.float32) / 255.0

        # Create RGB pseudo-color
        colored_heatmap = np.zeros((h, w, 3), dtype=np.float32)
        colored_heatmap[:, :, 0] = np.clip(heatmap_arr * 2.0, 0.0, 1.0) * 255
        colored_heatmap[:, :, 1] = np.clip((1.0 - np.abs(heatmap_arr - 0.5) * 2.0), 0.0, 1.0) * 255
        colored_heatmap[:, :, 2] = np.clip((1.0 - heatmap_arr) * 2.0, 0.0, 1.0) * 255

        orig_arr = np.array(img_pil, dtype=np.float32)
        blended = (1.0 - alpha) * orig_arr + alpha * colored_heatmap
        return np.clip(blended, 0, 255).astype(np.uint8)

    def close(self):
        """Removes PyTorch hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()

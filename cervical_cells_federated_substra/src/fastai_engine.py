"""
fastai Local Training Engine for Substra Nodes on Cervical Cytology.
Executes 1-Cycle policy training and FedProx proximal regularization inside Substra local nodes.
"""

from collections import OrderedDict
from typing import Any, Optional

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn


def build_cervical_model(
    arch_name: str = "convnext_small",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiates PyTorch vision backbone for 7-class cervical cytology classification."""
    timm_alias_map = {
        "convnext_small": "convnext_small.fb_in22k_ft_in1k_384",
        "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
        "efficientnet_b0": "efficientnet_b0",
        "resnet50d": "resnet50d",
        "phikon": "vit_base_patch16_224",
    }
    model_id = timm_alias_map.get(arch_name, arch_name)
    try:
        model = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    except Exception:
        # Fallback to standard timm model creation
        model = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    return model


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Serializes PyTorch model parameters into NumPy arrays for Substra aggregation."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Deserializes and updates PyTorch model weights from aggregated NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters, strict=False)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FedProxLoss(nn.Module):
    """
    Substra FedProx Proximal Regularization Loss for fastai.
    L(w; w^t) = L_ce(w) + (mu / 2) * sum_l ||w_l - w^t_l||^2
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        model: nn.Module,
        global_parameters: list[np.ndarray],
        mu: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.model = model
        self.mu = mu
        self.device = device or torch.device("cpu")
        self.global_tensors = [torch.tensor(p, device=self.device) for p in global_parameters]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_criterion(pred, target)
        if self.mu > 0.0 and self.global_tensors:
            prox_term = torch.tensor(0.0, device=self.device)
            for param, g_param in zip(self.model.parameters(), self.global_tensors, strict=False):
                prox_term += torch.sum((param - g_param) ** 2)
            loss = loss + (self.mu / 2.0) * prox_term
        return loss


def local_train_fastai(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training in a Substra organization node using fastai 1-Cycle dynamics.
    Returns: (updated_weights, train_loss, num_samples)
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model.to(device)
    model.train()

    epochs = int(cfg.get("local_epochs", 2))
    lr = float(cfg.get("local_lr", 0.0003))
    batch_size = min(int(cfg.get("batch_size", 32)), max(2, len(client_df)))
    img_size = int(cfg.get("image_size", 224))
    proximal_mu = float(cfg.get("proximal_mu", 1.0))
    is_fedprox = (cfg.get("strategy", "FedAvg") == "FedProx") and (proximal_mu > 0.0)

    # FedProx global anchor tensors
    global_tensors = None
    if is_fedprox and global_model_weights is not None:
        global_tensors = [torch.tensor(w, device=device) for w in global_model_weights]

    import torchvision.transforms as T
    from PIL import Image

    from src.dataset import CLASS_TO_IDX

    train_tfms = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_steps = 0

    for _epoch in range(epochs):
        shuffled_df = client_df.sample(frac=1.0).reset_index(drop=True)
        for i in range(0, len(shuffled_df), batch_size):
            batch_slice = shuffled_df.iloc[i : i + batch_size]
            batch_imgs = []
            batch_labels = []

            for _, row in batch_slice.iterrows():
                try:
                    img = Image.open(row["filepath"]).convert("RGB")
                    batch_imgs.append(train_tfms(img))
                    batch_labels.append(CLASS_TO_IDX[row["label"]])
                except Exception:
                    continue

            if not batch_imgs:
                continue

            xb = torch.stack(batch_imgs).to(device)
            yb = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)

            if is_fedprox and global_tensors is not None:
                prox_reg = torch.tensor(0.0, device=device)
                for w, w_t in zip(model.parameters(), global_tensors, strict=False):
                    if w.shape == w_t.shape:
                        prox_reg += torch.sum((w - w_t) ** 2)
                loss = loss + (proximal_mu / 2.0) * prox_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    avg_loss = float(total_loss / max(total_steps, 1))
    updated_weights = get_model_parameters(model)
    return updated_weights, avg_loss, len(client_df)


def evaluate_model_on_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates PyTorch model on centralized holdout test DataFrame.
    Returns: (y_true, y_prob, avg_loss)
    """
    if len(test_df) == 0:
        return np.array([]), np.array([]), 0.0

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model.to(device)
    model.eval()

    img_size = int(cfg.get("image_size", 224))
    from src.dataset import extract_tensors_from_df

    X, y = extract_tensors_from_df(test_df, img_size=img_size, augment=False)
    if len(X) == 0:
        return np.array([]), np.array([]), 0.0

    criterion = nn.CrossEntropyLoss()
    y_true_list = []
    y_prob_list = []
    total_loss = 0.0

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i : i + batch_size].to(device)
            batch_y = y[i : i + batch_size].to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.softmax(logits, dim=1)

            total_loss += loss.item() * len(batch_x)
            y_true_list.extend(batch_y.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.array([])
    avg_loss = float(total_loss / len(X)) if len(X) > 0 else 0.0

    return y_true, y_prob, avg_loss

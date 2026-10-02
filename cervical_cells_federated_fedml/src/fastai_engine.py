"""
fastai Vision Training Engine for Decentralized FedML Hospital Clients.
Supports vision backbones (ConvNeXt, ResNet), AdamW, and FedProx proximal regularization.
"""

from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn, optim

from src.dataset import CLASS_TO_IDX


class FastAIFedProxLoss(nn.Module):
    """
    Cross-entropy loss augmented with FedProx proximal regularization penalty:
    L_prox(w; w^t) = L_ce(w) + (mu / 2) * sum_l ||w_l - w^t_l||^2
    """

    def __init__(
        self,
        base_loss: nn.Module,
        model: nn.Module,
        global_anchor_tensors: list[torch.Tensor] | None = None,
        mu: float = 0.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.model = model
        self.global_anchor_tensors = global_anchor_tensors
        self.mu = mu

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(pred, target)
        if self.mu > 0.0 and self.global_anchor_tensors:
            prox_term = torch.tensor(0.0, device=pred.device)
            for p, g in zip(self.model.parameters(), self.global_anchor_tensors, strict=False):
                if p.shape == g.shape:
                    prox_term = prox_term + torch.sum((p - g.to(pred.device)) ** 2)
            loss = loss + (self.mu / 2.0) * prox_term
        return loss


def resolve_device(requested_device: str | None = None) -> torch.device:
    """Safely resolves device to CUDA if available, falling back to CPU."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested_device:
        try:
            d = torch.device(requested_device)
            if d.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return d
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cervical_model(
    arch_name: str = "convnext_small",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Constructs a PyTorch vision classifier backbone configured for 7-class cervical cytology.
    Uses timm for modern architectures (ConvNeXt, ViT, ResNet).
    """
    try:
        model = timm.create_model(
            arch_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    except Exception as e:
        print(f"[Model-Builder] Warning creating '{arch_name}' via timm: {e}. Falling back to resnet34.")
        model = timm.create_model("resnet34", pretrained=pretrained, num_classes=num_classes)

    return model


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Serializes all model parameters into a list of NumPy arrays."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Injects a list of NumPy arrays into the model parameters."""
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), parameters, strict=False):
            param.copy_(torch.from_numpy(new_val).to(param.device))


def local_train_fastai(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training loop for an individual hospital node.
    Supports AdamW optimization and FedProx proximal regularization.
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    batch_size = min(int(cfg.get("batch_size", 16)), max(2, len(client_df)))
    local_epochs = int(cfg.get("local_epochs", 2))
    local_lr = float(cfg.get("local_lr", 0.0003))
    device = resolve_device(cfg.get("device"))
    strategy = cfg.get("strategy", "FedAvg")
    mu = float(cfg.get("proximal_mu", 1.0)) if strategy == "FedProx" else 0.0
    img_size = int(cfg.get("image_size", 224))

    model.to(device)
    model.train()

    # FedProx global anchor tensors
    global_tensors = None
    if strategy == "FedProx" and global_model_weights is not None and mu > 0.0:
        global_tensors = [torch.tensor(w, device=device) for w in global_model_weights]

    train_tfms = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    optimizer = optim.AdamW(model.parameters(), lr=local_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_steps = 0

    for _epoch in range(local_epochs):
        shuffled_df = client_df.sample(frac=1.0).reset_index(drop=True)
        for i in range(0, len(shuffled_df), batch_size):
            batch_slice = shuffled_df.iloc[i : i + batch_size]
            batch_imgs = []
            batch_labels = []

            for _, row in batch_slice.iterrows():
                path_str = row.get("image_path", row.get("filepath", ""))
                try:
                    img = Image.open(path_str).convert("RGB")
                    batch_imgs.append(train_tfms(img))
                    lbl = row["label"]
                    lbl_idx = int(lbl) if isinstance(lbl, (int, np.integer)) else CLASS_TO_IDX[str(lbl)]
                    batch_labels.append(lbl_idx)
                except Exception:
                    continue

            if not batch_imgs:
                continue

            xb = torch.stack(batch_imgs).to(device)
            yb = torch.tensor(batch_labels, dtype=torch.long).to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)

            if strategy == "FedProx" and global_tensors is not None and mu > 0.0:
                prox_reg = torch.tensor(0.0, device=device)
                for w, w_t in zip(model.parameters(), global_tensors, strict=False):
                    if w.shape == w_t.shape:
                        prox_reg += torch.sum((w - w_t) ** 2)
                loss = loss + (mu / 2.0) * prox_reg

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    updated_weights = get_model_parameters(model)
    return updated_weights, avg_loss, len(client_df)


def evaluate_model_on_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates model predictions on the centralized holdout test partition.
    Returns (y_true, y_prob, avg_loss).
    """
    device = resolve_device(cfg.get("device"))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0

    batch_size = int(cfg.get("batch_size", 32))
    loss_fn = nn.CrossEntropyLoss()

    for start_idx in range(0, len(test_df), batch_size):
        chunk_df = test_df.iloc[start_idx : start_idx + batch_size]
        from src.dataset import extract_tensors_from_df

        x_b, y_b = extract_tensors_from_df(chunk_df, img_size=224)
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        with torch.no_grad():
            logits = model(x_b)
            loss = loss_fn(logits, y_b)
            probs = F.softmax(logits, dim=-1)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(y_b.cpu().numpy())
        total_loss += float(loss.item()) * len(y_b)
        num_samples += len(y_b)

    y_prob = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, 7))
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=int)
    avg_loss = total_loss / max(1, num_samples)

    return y_true, y_prob, avg_loss

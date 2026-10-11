"""
fastai Vision Training Engine for Cervical Cytology.
Supports ConvNeXt/ResNet vision backbones, AdamW/1-Cycle loop,
and FedProx proximal regularization penalty.
"""

from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms

from src.dataset import CLASS_TO_IDX


def resolve_device(device_str: str | None = None) -> torch.device:
    """Safely resolves execution device, falling back to CPU if CUDA is not available."""
    if device_str is not None:
        d = torch.device(device_str)
        if d.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return d
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_cervical_model(
    arch_name: str = "convnext_small",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Builds a vision model backbone via timm with custom classification head."""
    try:
        model = timm.create_model(
            arch_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    except (ValueError, RuntimeError) as e:
        print(f"[Model-Builder] Warning creating '{arch_name}' via timm: {e}. Falling back to resnet34.")
        model = timm.create_model("resnet34", pretrained=pretrained, num_classes=num_classes)

    return model


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extracts model weights as a list of NumPy arrays."""
    return [param.detach().cpu().numpy().copy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Loads weights from a list of NumPy arrays into a PyTorch model."""
    with torch.no_grad():
        for param, new_val in zip(model.parameters(), parameters, strict=False):
            param.copy_(torch.from_numpy(new_val).to(param.device))


def local_train_fastai(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
    preprocessor: Any = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training loop using fastai/PyTorch on an individual hospital silo.
    Applies AdamW optimizer and optional FedProx proximal penalty constraint.
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    device = resolve_device(cfg.get("device"))
    model = model.to(device)
    model.train()

    batch_size = int(cfg.get("batch_size", 16))
    local_epochs = int(cfg.get("local_epochs", 2))
    local_lr = float(cfg.get("local_lr", 0.0003))
    strategy = cfg.get("strategy", "FedAvg")
    mu = float(cfg.get("proximal_mu", 1.0)) if strategy == "FedProx" else 0.0

    global_tensors = None
    if strategy == "FedProx" and global_model_weights is not None:
        global_tensors = [torch.from_numpy(w).to(device) for w in global_model_weights]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    optimizer = optim.AdamW(model.parameters(), lr=local_lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_steps = 0

    num_samples = len(client_df)
    indices = np.arange(num_samples)

    for _ in range(local_epochs):
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_idxs = indices[start_idx : start_idx + batch_size]
            batch_imgs = []
            batch_labels = []

            for idx in batch_idxs:
                row = client_df.iloc[idx]
                img_p = row["image_path"] if "image_path" in row else row["filepath"]
                lbl = row["label"]
                try:
                    with Image.open(img_p) as pil_img:
                        rgb = pil_img.convert("RGB")
                        if preprocessor is not None:
                            if hasattr(preprocessor, "preprocess_pil"):
                                rgb = preprocessor.preprocess_pil(rgb)
                            else:
                                rgb = preprocessor(rgb)
                        t = transform(rgb)
                        batch_imgs.append(t)
                        lbl_idx = int(lbl) if isinstance(lbl, (int, np.integer)) else CLASS_TO_IDX[str(lbl)]
                        batch_labels.append(lbl_idx)
                except (OSError, RuntimeError, ValueError):
                    continue

            if not batch_imgs:
                continue

            x_batch = torch.stack(batch_imgs).to(device)
            y_batch = torch.tensor(batch_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if strategy == "FedProx" and global_tensors is not None:
                prox_penalty = torch.tensor(0.0, device=device)
                for w, w_t in zip(model.parameters(), global_tensors, strict=False):
                    prox_penalty += ((w - w_t) ** 2).sum()
                loss = loss + (mu / 2.0) * prox_penalty

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
    preprocessor: Any = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates model on centralized holdout test set.
    Returns (y_true, y_prob, avg_loss).
    """
    device = resolve_device(cfg.get("device"))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch_size = int(cfg.get("batch_size", 16))
    criterion = nn.CrossEntropyLoss()

    all_preds: list[np.ndarray] = []
    all_targets: list[int] = []
    total_loss = 0.0
    total_batches = 0

    indices = np.arange(len(test_df))

    with torch.no_grad():
        for start_idx in range(0, len(test_df), batch_size):
            batch_idxs = indices[start_idx : start_idx + batch_size]
            batch_imgs = []
            batch_labels = []

            for idx in batch_idxs:
                row = test_df.iloc[idx]
                img_p = row["image_path"] if "image_path" in row else row["filepath"]
                lbl = row["label"]
                try:
                    with Image.open(img_p) as pil_img:
                        rgb = pil_img.convert("RGB")
                        if preprocessor is not None:
                            if hasattr(preprocessor, "preprocess_pil"):
                                rgb = preprocessor.preprocess_pil(rgb)
                            else:
                                rgb = preprocessor(rgb)
                        t = transform(rgb)
                        batch_imgs.append(t)
                        lbl_idx = int(lbl) if isinstance(lbl, (int, np.integer)) else CLASS_TO_IDX[str(lbl)]
                        batch_labels.append(lbl_idx)
                except (OSError, RuntimeError, ValueError):
                    continue

            if not batch_imgs:
                continue

            x_batch = torch.stack(batch_imgs).to(device)
            y_batch = torch.tensor(batch_labels, dtype=torch.long, device=device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.append(probs)
            all_targets.extend(batch_labels)
            total_loss += float(loss.item())
            total_batches += 1

    if not all_preds:
        return np.array([]), np.array([]), 0.0

    y_prob = np.concatenate(all_preds, axis=0)
    y_true = np.array(all_targets)
    avg_loss = total_loss / max(1, total_batches)

    return y_true, y_prob, avg_loss

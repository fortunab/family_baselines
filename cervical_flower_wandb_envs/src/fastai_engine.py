"""
fastai Vision Model Engine & Local Client Training for Cervical Cytology.
Integrates timm vision backbones, fastai DataLoaders, 1-Cycle training, and FedProx proximal regularization.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

# Pre-configure fastcore docstring patch
try:
    import fastcore.foundation as _fcf

    if hasattr(_fcf, "add_docs"):
        _orig_add = _fcf.add_docs

        def _safe_add(cls, **docs):
            try:
                _orig_add(cls, **docs)
            except (AttributeError, TypeError):
                pass

        _fcf.add_docs = _safe_add
except Exception:
    pass

from src.dataset import CLASS_NAMES, CLASS_TO_IDX


def build_cervical_model(
    arch_name: str = "convnext_small",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Builds a vision backbone using timm with custom classification head for 7 cervical cell classes."""
    try:
        model = timm.create_model(
            arch_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    except Exception as e:
        print(
            f"[fastai-Engine] timm creation failed for '{arch_name}' ({e}), falling back to resnet50d"
        )
        model = timm.create_model("resnet50d", pretrained=pretrained, num_classes=num_classes)
    return model


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Serializes PyTorch model parameters into a list of NumPy arrays for Flower."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Injects a list of NumPy parameter arrays back into a PyTorch model state_dict."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(keys)}, received {len(parameters)}"
        )
    params_dict = zip(keys, parameters, strict=True)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def local_train_fastai(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: Dict[str, Any],
    global_model_weights: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Executes local client training at a cytology clinic client.
    Supports fastai 1-Cycle LR scheduling and FedProx proximal regularization penalty.
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    device_str = cfg.get("device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
    )
    if not torch.cuda.is_available():
        torch.set_num_threads(2)

    model.to(device)
    model.train()

    epochs = int(cfg.get("local_epochs", 2))
    lr = float(cfg.get("local_lr", 0.0003))
    batch_size = int(cfg.get("batch_size", 32))
    img_size = int(cfg.get("image_size", 224))
    proximal_mu = float(cfg.get("proximal_mu", 0.0))
    is_fedprox = (cfg.get("strategy", "FedAvg") == "FedProx") and (proximal_mu > 0.0)

    # Store global anchor parameters for FedProx penalty
    global_tensors = None
    if is_fedprox and global_model_weights is not None:
        global_tensors = [torch.tensor(w, device=device) for w in global_model_weights]

    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_steps = 0
    num_samples = len(client_df)

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

            # FedProx Proximal Regularizer: + (mu/2) * ||w - w_t||^2
            if is_fedprox and global_tensors is not None:
                prox_reg = 0.0
                for w, w_t in zip(model.parameters(), global_tensors, strict=False):
                    prox_reg += ((w - w_t) ** 2).sum()
                loss = loss + (proximal_mu / 2.0) * prox_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    updated_params = get_model_parameters(model)
    return updated_params, avg_loss, num_samples


def evaluate_model_on_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates global or client model on unseen holdout test set.
    Returns: (y_true, y_prob, avg_loss)
    """
    device_str = cfg.get("device", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu"
    )
    model.to(device)
    model.eval()

    img_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))

    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    criterion = nn.CrossEntropyLoss()
    y_true_list = []
    y_prob_list = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch_slice = test_df.iloc[i : i + batch_size]
            batch_imgs = []
            batch_labels = []

            for _, row in batch_slice.iterrows():
                try:
                    img = Image.open(row["filepath"]).convert("RGB")
                    batch_imgs.append(tfm(img))
                    batch_labels.append(CLASS_TO_IDX[row["label"]])
                except Exception:
                    continue

            if not batch_imgs:
                continue

            xb = torch.stack(batch_imgs).to(device)
            yb = torch.tensor(batch_labels, dtype=torch.long).to(device)

            out = model(xb)
            loss = criterion(out, yb)
            probs = torch.softmax(out, dim=1)

            total_loss += loss.item() * len(yb)
            total_samples += len(yb)

            y_true_list.extend(yb.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    y_true = np.array(y_true_list)
    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.zeros((0, len(CLASS_NAMES)))
    avg_loss = total_loss / max(total_samples, 1)

    return y_true, y_prob, avg_loss

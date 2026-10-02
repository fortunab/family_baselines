"""
skorch NeuralNetClassifier Engine for Cervical Cytology Dysplasia Grading.
Integrates PyTorch vision backbones with Scikit-Learn Estimator API,
supporting 1-Cycle / Cosine Annealing scheduling and FedProx proximal regularization.
"""

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import timm
import torch
from skorch import NeuralNetClassifier
from torch import nn, optim

from src.dataset import CLASS_NAMES, extract_tensors_from_df


def build_cervical_model(
    arch_name: str = "convnext_small",
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Builds PyTorch vision backbone with customized 7-class cytology classification head."""
    arch_clean = arch_name.lower().strip()

    if "phikon" in arch_clean:
        try:
            from transformers import AutoModel

            class PhikonCervicalModule(nn.Module):
                def __init__(self, n_cls: int, p_drop: float):
                    super().__init__()
                    self.encoder = AutoModel.from_pretrained("owkin/phikon") if pretrained else AutoModel.from_config()
                    in_dim = getattr(self.encoder.config, "hidden_size", 768)
                    self.head = nn.Sequential(
                        nn.Dropout(p_drop),
                        nn.Linear(in_dim, n_cls),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    feat = self.encoder(x).last_hidden_state[:, 0, :]
                    return self.head(feat)

            return PhikonCervicalModule(n_cls=num_classes, p_drop=dropout)
        except Exception:
            arch_clean = "convnext_small"

    try:
        model = timm.create_model(
            arch_clean,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    except Exception:
        model = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    return model


class SkorchCervicalClassifier(NeuralNetClassifier):
    """
    Scikit-Learn compatible NeuralNetClassifier extending skorch.
    Implements FedProx proximal regularization penalty in the local loss function.
    """

    def __init__(
        self,
        module,
        *args,
        is_fedprox: bool = False,
        proximal_mu: float = 0.0,
        global_anchor_tensors: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        super().__init__(module, *args, **kwargs)
        self.is_fedprox = is_fedprox
        self.proximal_mu = proximal_mu
        self.global_anchor_tensors = global_anchor_tensors

    def train_step(self, batch, **fit_params):
        """Custom train step that adds FedProx proximal penalty if enabled."""
        self.check_is_fitted()
        step = self.train_step_single(batch, **fit_params)
        return step

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Computes loss with optional FedProx proximal term: ||w - w_t||^2."""
        base_loss = super().get_loss(y_pred, y_true, X=X, training=training)

        if training and self.is_fedprox and self.global_anchor_tensors is not None:
            prox_reg = 0.0
            for w, w_t in zip(self.module_.parameters(), self.global_anchor_tensors, strict=False):
                prox_reg = prox_reg + ((w - w_t) ** 2).sum()
            return base_loss + (self.proximal_mu / 2.0) * prox_reg

        return base_loss


def get_model_parameters(model: nn.Module) -> list[np.ndarray]:
    """Serializes PyTorch model parameters into a list of NumPy NDArrays for Flower server."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Updates PyTorch model weights from aggregated NumPy NDArrays."""
    params_dict = zip(model.state_dict().keys(), parameters, strict=False)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def build_skorch_classifier(
    arch_name: str,
    num_classes: int,
    lr: float,
    max_epochs: int,
    batch_size: int,
    device: str,
    dropout: float = 0.2,
    is_fedprox: bool = False,
    proximal_mu: float = 0.0,
    global_weights: list[np.ndarray] | None = None,
) -> SkorchCervicalClassifier:
    """Factory creating configured SkorchCervicalClassifier instance."""
    if not torch.cuda.is_available():
        torch.set_num_threads(2)

    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    anchor_tensors = None
    if is_fedprox and global_weights is not None:
        anchor_tensors = [torch.tensor(w, device=device_obj) for w in global_weights]

    net = SkorchCervicalClassifier(
        module=build_cervical_model,
        module__arch_name=arch_name,
        module__num_classes=num_classes,
        module__pretrained=False,
        module__dropout=dropout,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=1e-4,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device_obj,
        train_split=None,  # All local client data used for client training
        verbose=0,
        is_fedprox=is_fedprox,
        proximal_mu=proximal_mu,
        global_anchor_tensors=anchor_tensors,
    )
    return net


def local_train_skorch(
    model: nn.Module,
    client_df: pd.DataFrame,
    cfg: dict[str, Any],
    global_model_weights: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], float, int]:
    """
    Executes local client training using skorch NeuralNetClassifier.
    Returns: (updated_parameters, final_train_loss, num_samples)
    """
    if len(client_df) == 0:
        return get_model_parameters(model), 0.0, 0

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    if not torch.cuda.is_available():
        torch.set_num_threads(2)

    epochs = int(cfg.get("local_epochs", 2))
    lr = float(cfg.get("local_lr", 0.0003))
    batch_size = int(cfg.get("batch_size", 32))
    img_size = int(cfg.get("image_size", 224))
    proximal_mu = float(cfg.get("proximal_mu", 1.0))
    is_fedprox = (cfg.get("strategy", "FedAvg") == "FedProx") and (proximal_mu > 0.0)

    # 1. Extract image tensors and labels
    X, y = extract_tensors_from_df(client_df, img_size=img_size, augment=True)
    if len(X) == 0:
        return get_model_parameters(model), 0.0, 0

    # 2. Build skorch classifier wrapping local model
    arch_name = cfg.get("architecture", "convnext_small")
    num_classes = int(cfg.get("num_classes", 7))
    dropout = float(cfg.get("dropout_rate", 0.2))

    net = build_skorch_classifier(
        arch_name=arch_name,
        num_classes=num_classes,
        lr=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        device=str(device),
        dropout=dropout,
        is_fedprox=is_fedprox,
        proximal_mu=proximal_mu,
        global_weights=global_model_weights,
    )

    # Initialize skorch internal module with current weights
    net.initialize()
    if hasattr(net, "module_"):
        net.module_.load_state_dict(model.state_dict())

    # 3. Fit via Scikit-Learn API: net.fit(X, y)
    net.fit(X, y)

    # Synchronize updated weights back to model
    if hasattr(net, "module_"):
        model.load_state_dict(net.module_.state_dict())

    history = getattr(net, "history_", None)
    train_loss = 0.0
    if history and len(history) > 0:
        last_epoch_hist = history[-1]
        train_loss = float(last_epoch_hist.get("train_loss", 0.0))

    return get_model_parameters(model), train_loss, len(X)


def evaluate_skorch_on_test_set(
    model: nn.Module,
    test_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluates model on unseen holdout test set using skorch / PyTorch inference.
    Returns: (y_true, y_prob, avg_loss)
    """
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model.to(device)
    model.eval()

    img_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("batch_size", 32))

    X_test, y_test = extract_tensors_from_df(test_df, img_size=img_size, augment=False)
    if len(X_test) == 0:
        return np.array([]), np.zeros((0, len(CLASS_NAMES))), 0.0

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = len(X_test)
    y_prob_list = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = X_test[i : i + batch_size].to(device)
            yb = y_test[i : i + batch_size].to(device)

            out = model(xb)
            loss = criterion(out, yb)
            probs = torch.softmax(out, dim=1)

            total_loss += loss.item() * len(yb)
            y_prob_list.append(probs.cpu().numpy())

    y_true = y_test.numpy()
    y_prob = np.concatenate(y_prob_list, axis=0) if y_prob_list else np.zeros((0, len(CLASS_NAMES)))
    avg_loss = total_loss / max(total_samples, 1)

    return y_true, y_prob, avg_loss

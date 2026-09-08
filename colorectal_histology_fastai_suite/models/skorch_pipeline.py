"""
skorch Scikit-Learn NeuralNetClassifier Pipeline for Colorectal Histology.
Wraps PyTorch vision foundation models with Scikit-Learn interface.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from core.dataset_loader import get_numpy_tensors
from core.metrics_evaluator import evaluate_multiclass_predictions, print_metrics_report, CLASS_NAMES


class HistologyVisionModule(nn.Module):
    def __init__(self, backbone_name: str = "resnet50d", num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.model = None
        self._init_backbone(num_classes, pretrained)

    def _init_backbone(self, num_classes: int, pretrained: bool):
        try:
            import timm
            self.model = timm.create_model(self.backbone_name, pretrained=pretrained, num_classes=num_classes)
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            m = resnet50(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
            self.model = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_and_eval_skorch(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tracker: Any
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    import skorch
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring

    backbone_name = cfg.get("backbone", "resnet50d")
    img_size = int(cfg.get("image_size", 224))
    bs = int(cfg.get("batch_size", 16))
    max_epochs = int(cfg.get("max_epochs", cfg.get("epochs", 8)))
    lr = float(cfg.get("lr", cfg.get("learning_rate", 0.001)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n[skorch] Extracting NumPy image tensors (ImageSize={img_size})...")
    X_train, y_train = get_numpy_tensors(train_df, image_size=img_size)
    X_val, y_val = get_numpy_tensors(val_df, image_size=img_size)
    X_test, y_test = get_numpy_tensors(test_df, image_size=img_size)

    print(f"[skorch] Tensors: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Set up skorch callbacks
    callbacks = [
        LRScheduler(policy=optim.lr_scheduler.CosineAnnealingLR, T_max=max_epochs),
        EpochScoring(scoring='accuracy', name='val_acc', lower_is_better=False),
        EarlyStopping(patience=int(cfg.get("early_stopping_patience", 5)), monitor='val_acc', lower_is_better=False)
    ]

    from skorch.helper import predefined_split
    val_dataset = skorch.dataset.Dataset(X_val, y_val)

    print(f"[skorch] Initializing NeuralNetClassifier (Backbone={backbone_name}, MaxEpochs={max_epochs}, Device={device})...")
    net = NeuralNetClassifier(
        module=HistologyVisionModule,
        module__backbone_name=backbone_name,
        module__num_classes=8,
        module__pretrained=bool(cfg.get("pretrained", True)),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=float(cfg.get("weight_decay", 1e-4)),
        batch_size=bs,
        max_epochs=max_epochs,
        device=device,
        callbacks=callbacks,
        train_split=predefined_split(val_dataset),
        verbose=1
    )

    print(f"[skorch] Training model with Scikit-Learn .fit() API...")
    net.fit(X_train, y_train)

    print(f"[skorch] Evaluating on unseen holdout test set with .predict_proba()...")
    y_prob = net.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    test_metrics = evaluate_multiclass_predictions(y_test, y_pred, y_prob)
    test_metrics["Framework"] = "skorch"
    test_metrics["Backbone"] = backbone_name

    print_metrics_report(test_metrics, backbone_name, framework="skorch", seed=cfg.get("seed"))

    return test_metrics, y_test, y_pred, y_prob

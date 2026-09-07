"""
Pure skorch NeuralNetClassifier Engine for Pathology Foundation Models.
Wraps Pathology Foundation backbones in Scikit-Learn Estimator interface.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring, LRScheduler
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import extract_numpy_tensors
from src.foundation_models import PathologyFoundationClassifier
from src.wandb_tracker import SkorchWandbCallback, WandbExperimentTracker


def train_skorch_foundation_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tracker: WandbExperimentTracker,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    backbone_name = cfg.get("backbone", "owkin/phikon")
    img_size = int(cfg.get("image_size", 224))
    bs = int(cfg.get("batch_size", 16))
    max_epochs = int(cfg.get("epochs", cfg.get("max_epochs", 8)))
    lr = float(cfg.get("learning_rate", cfg.get("lr", 0.0003)))
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu"

    print(f"\n[skorch-Foundation] Extracting NumPy image tensors (ImageSize={img_size})...")
    X_train, y_train = extract_numpy_tensors(train_df, img_size=img_size)
    X_val, y_val = extract_numpy_tensors(val_df, img_size=img_size)
    X_test, y_test = extract_numpy_tensors(test_df, img_size=img_size)

    print(
        f"[skorch-Foundation] Tensors: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    val_dataset = skorch.dataset.Dataset(X_val, y_val)

    callbacks = [
        EpochScoring(scoring="accuracy", name="val_acc", lower_is_better=False),
        LRScheduler(policy=CosineAnnealingLR, T_max=max_epochs),
        EarlyStopping(
            patience=int(cfg.get("early_stopping_patience", 5)),
            monitor="val_acc",
            lower_is_better=False,
        ),
        SkorchWandbCallback(tracker=tracker),
    ]

    print(
        f"[skorch-Foundation] Initializing NeuralNetClassifier (Backbone={backbone_name}, MaxEpochs={max_epochs}, Device={device})..."
    )
    net = NeuralNetClassifier(
        module=PathologyFoundationClassifier,
        module__backbone_name=backbone_name,
        module__num_classes=8,
        module__embedding_dim=cfg.get("embedding_dim"),
        module__pretrained=bool(cfg.get("pretrained", True)),
        module__model_type=cfg.get("model_type", "huggingface"),
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=float(cfg.get("weight_decay", 1e-4)),
        batch_size=bs,
        max_epochs=max_epochs,
        device=device,
        callbacks=callbacks,
        train_split=predefined_split(val_dataset),
        verbose=1,
    )

    print("[skorch-Foundation] Training model with Scikit-Learn .fit() API...")
    net.fit(X_train, y_train)

    print("[skorch-Foundation] Evaluating on unseen holdout test set with .predict_proba()...")
    y_prob = net.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = y_test

    from src.evaluator import evaluate_multiclass_metrics, print_evaluation_report

    test_metrics = evaluate_multiclass_metrics(y_true, y_pred, y_prob)
    test_metrics["Framework"] = "skorch"
    test_metrics["Backbone"] = backbone_name

    print_evaluation_report(test_metrics, backbone_name, seed=cfg.get("seed"))

    return test_metrics, y_true, y_pred, y_prob

"""
fastai Vision Foundation Pipeline for Colorectal Histology.
Uses fastai.vision.all, vision_learner, fine_tune protocol, and MLOps Callbacks.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

from core.dataset_loader import CLASS_NAMES, CLASS_TO_IDX


def train_and_eval_fastai(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tracker: Any
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    import fastai.vision.all as fa

    backbone_name = cfg.get("backbone", "convnext_base")
    img_size = int(cfg.get("image_size", 224))
    bs = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 5))
    freeze_epochs = int(cfg.get("freeze_epochs", 1))
    lr = float(cfg.get("learning_rate", 0.001))
    wd = float(cfg.get("weight_decay", 0.01))

    print(f"\n[fastai] Setting up DataBlock (ImageSize={img_size}, BatchSize={bs})...")

    # Combine train & val into a single df with is_valid column
    train_df_c = train_df.copy()
    train_df_c["is_valid"] = False
    val_df_c = val_df.copy()
    val_df_c["is_valid"] = True

    combined_df = pd.concat([train_df_c, val_df_c], ignore_index=True)

    # Item & Batch transforms
    item_tfms = [fa.Resize(img_size, method='squish')]
    batch_tfms = [
        fa.Rotate(max_deg=float(cfg.get("max_rotate", 15.0))),
        fa.Zoom(max_zoom=float(cfg.get("max_zoom", 1.1))),
        fa.Normalize.from_stats(*fa.imagenet_stats)
    ]

    dblock = fa.DataBlock(
        blocks=(fa.ImageBlock, fa.CategoryBlock(vocab=CLASS_NAMES)),
        get_x=fa.ColReader("filepath"),
        get_y=fa.ColReader("label"),
        splitter=fa.ColSplitter("is_valid"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(combined_df, bs=bs)

    print(f"[fastai] Initializing vision_learner with backbone: '{backbone_name}'...")
    
    # Try creating timm / fastai vision learner
    try:
        learn = fa.vision_learner(
            dls,
            backbone_name,
            metrics=[fa.accuracy, fa.BalancedAccuracy(), fa.F1Score(average='macro')],
            wd=wd,
            pretrained=bool(cfg.get("pretrained", True))
        )
    except Exception as e:
        print(f"[fastai] Direct backbone load notice ({e}), defaulting to resnet50...")
        learn = fa.vision_learner(
            dls,
            fa.resnet50,
            metrics=[fa.accuracy, fa.BalancedAccuracy(), fa.F1Score(average='macro')],
            wd=wd
        )

    # Attach W&B callback if requested
    if tracker.backend in ("wandb", "weights_and_biases", "w&b") and tracker.wandb_run is not None:
        try:
            from fastai.callback.wandb import WandbCallback
            learn.add_cb(WandbCallback(log_preds=False))
            print("[fastai] Attached fastai WandbCallback.")
        except Exception as e:
            print(f"[fastai] WandbCallback notice ({e}).")

    print(f"[fastai] Training {backbone_name} with fine_tune({epochs}, base_lr={lr}, freeze_epochs={freeze_epochs})...")
    learn.fine_tune(epochs=epochs, base_lr=lr, freeze_epochs=freeze_epochs)

    # Evaluate on Unseen 15% Holdout Test Set
    print(f"[fastai] Evaluating on unseen holdout test set ({len(test_df)} samples)...")
    test_dl = learn.dls.test_dl(test_df["filepath"].tolist(), with_labels=False)
    preds, _ = learn.get_preds(dl=test_dl)

    y_prob = preds.numpy()
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"].tolist()], dtype=np.int64)

    from core.metrics_evaluator import evaluate_multiclass_predictions, print_metrics_report
    test_metrics = evaluate_multiclass_predictions(y_true, y_pred, y_prob)
    test_metrics["Framework"] = "fastai"
    test_metrics["Backbone"] = backbone_name

    print_metrics_report(test_metrics, backbone_name, framework="fastai", seed=cfg.get("seed"))

    return test_metrics, y_true, y_pred, y_prob

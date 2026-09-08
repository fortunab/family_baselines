"""
Pure fastai Vision Foundation Training Engine.
Uses fastai.vision.all, vision_learner, fine_tune protocol, and WandbCallback.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Any, Dict, Tuple

# Compatibility patch for fastcore & PyTorch / Python 3.12+ (read-only __doc__ on C-extension functions)
try:
    import fastcore.foundation

    def _safe_add_docs(cls, cls_doc=None, **docs):
        if cls_doc is not None:
            try:
                cls.__doc__ = cls_doc
            except (AttributeError, TypeError):
                pass
        for k, v in docs.items():
            try:
                f = getattr(cls, k)
                f.__doc__ = v
            except (AttributeError, TypeError):
                pass

    fastcore.foundation.add_docs = _safe_add_docs
except Exception:
    pass

import torch
import fastai.vision.all as fa
import numpy as np
import pandas as pd
import timm

from src.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.wandb_tracker import WandbExperimentTracker


def train_fastai_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    tracker: WandbExperimentTracker,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    backbone_name = cfg.get("backbone", "convnext_base")
    img_size = int(cfg.get("image_size", 224))
    bs = int(cfg.get("batch_size", 16))
    epochs = int(cfg.get("epochs", 8))
    freeze_epochs = int(cfg.get("freeze_epochs", 1))
    lr = float(cfg.get("learning_rate", 0.0005))
    wd = float(cfg.get("weight_decay", 0.01))

    print(f"\n[fastai] Setting up DataBlock (ImageSize={img_size}, BatchSize={bs})...")

    train_df_c = train_df.copy()
    train_df_c["is_valid"] = False
    val_df_c = val_df.copy()
    val_df_c["is_valid"] = True

    combined_df = pd.concat([train_df_c, val_df_c], ignore_index=True)

    item_tfms = [fa.Resize(img_size, method="squish")]
    batch_tfms = [
        fa.Rotate(max_deg=float(cfg.get("max_rotate", 15.0))),
        fa.Zoom(max_zoom=float(cfg.get("max_zoom", 1.1))),
        fa.Normalize.from_stats(*fa.imagenet_stats),
    ]

    dblock = fa.DataBlock(
        blocks=(fa.ImageBlock, fa.CategoryBlock(vocab=CLASS_NAMES)),
        get_x=fa.ColReader("filepath"),
        get_y=fa.ColReader("label"),
        splitter=fa.ColSplitter("is_valid"),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(combined_df, bs=bs, num_workers=0)

    print(f"[fastai] Instantiating vision_learner with backbone: '{backbone_name}'...")
    try:
        learn = fa.vision_learner(
            dls,
            backbone_name,
            metrics=[fa.accuracy],
            wd=wd,
            pretrained=bool(cfg.get("pretrained", True)),
        )
    except Exception as e:
        print(
            f"[fastai] Notice loading '{backbone_name}' directly ({e}), falling back to resnet50..."
        )
        learn = fa.vision_learner(
            dls,
            fa.resnet50,
            metrics=[fa.accuracy],
            wd=wd,
        )

    # Attach fastai WandbCallback if W&B run is active
    if tracker.wandb_run is not None:
        try:
            from fastai.callback.wandb import WandbCallback

            learn.add_cb(WandbCallback(log_preds=bool(cfg.get("log_preds", False))))
            print("[fastai] Attached fastai WandbCallback for live telemetry logging.")
        except Exception as e:
            print(f"[fastai] WandbCallback attachment notice: {e}")

    print(
        f"[fastai] Fine-tuning {backbone_name} (epochs={epochs}, base_lr={lr}, freeze_epochs={freeze_epochs})..."
    )
    learn.fine_tune(epochs=epochs, base_lr=lr, freeze_epochs=freeze_epochs)

    # Evaluate on unseen holdout test set
    print(
        f"[fastai] Generating predictions on unseen holdout test set ({len(test_df)} samples)..."
    )
    test_dl = learn.dls.test_dl(test_df["filepath"].tolist(), with_labels=False, num_workers=0)
    preds, _ = learn.get_preds(dl=test_dl)

    y_prob = preds.numpy()
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.array([CLASS_TO_IDX[lbl] for lbl in test_df["label"].tolist()], dtype=np.int64)

    from src.evaluator import evaluate_multiclass_metrics, print_evaluation_report

    test_metrics = evaluate_multiclass_metrics(y_true, y_pred, y_prob)
    test_metrics["Framework"] = "fastai"
    test_metrics["Backbone"] = backbone_name

    print_evaluation_report(test_metrics, backbone_name, seed=cfg.get("seed"))

    return test_metrics, y_true, y_pred, y_prob

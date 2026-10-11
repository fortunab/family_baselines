"""Foundational Vision Model Training and Evaluation Engine.

Evaluates 4 Vision Foundation Backbones:
1. DenseNet-121 (Radiology Standard)
2. ResNet-50 (Deep Residual Standard)
3. ConvNeXt-Small (Modern 2020s ConvNet)
4. EfficientNet-B0 (Compound Scaled MBConv)

Operates on Grayscaled Alzheimer MRI scans, computing full clinical metrics and resource efficiency.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fastai.vision.all import (
    CategoryBlock,
    ColReader,
    DataBlock,
    ImageBlock,
    IndexSplitter,
    Resize,
    accuracy,
    vision_learner,
)
from torchvision import models

from src.evaluator import evaluate_clinical_metrics

FOUNDATION_ARCH_MAP = {
    "densenet121": models.densenet121,
    "resnet50": models.resnet50,
    "convnext_small": models.convnext_small,
    "efficientnet_b0": models.efficientnet_b0,
}


def build_fastai_foundation_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str],
    image_size: int = 224,
    batch_size: int = 16,
):
    """Builds fastai DataLoaders with strict train / test index splitting."""
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    val_indices = list(range(len(train_df), len(full_df)))

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=class_names)),
        get_x=ColReader("image_path"),
        get_y=ColReader("class_name"),
        splitter=IndexSplitter(val_indices),
        item_tfms=Resize(image_size),
    )

    dls = dblock.dataloaders(full_df, bs=batch_size)
    return dls


def count_parameters(model: torch.nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval_foundation_model(
    arch_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str],
    epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 16,
    image_size: int = 224,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Fine-tunes a vision foundation backbone on grayscaled MRI scans and calculates clinical metrics."""
    if arch_name not in FOUNDATION_ARCH_MAP:
        raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(FOUNDATION_ARCH_MAP.keys())}")

    arch_fn = FOUNDATION_ARCH_MAP[arch_name]
    dls = build_fastai_foundation_dataloaders(
        train_df=train_df,
        test_df=test_df,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
    )

    learner = vision_learner(
        dls,
        arch_fn,
        metrics=[accuracy],
        pretrained=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        learner.model.to("cuda")
    else:
        learner.model.to("cpu")

    num_params = count_parameters(learner.model)

    # Train
    t_start = time.time()
    learner.fit_one_cycle(epochs, lr_max=lr)
    t_train = time.time() - t_start

    # Evaluate on test split (the validation split of the dls)
    t_inf_start = time.time()
    preds, targets = learner.get_preds(ds_idx=1)
    t_inf = time.time() - t_inf_start

    probs = preds.numpy()
    pred_labels = np.argmax(probs, axis=1)
    true_labels = targets.numpy()
    inf_latency_ms = (t_inf / len(true_labels)) * 1000.0

    metrics = evaluate_clinical_metrics(
        y_true=true_labels,
        y_pred=pred_labels,
        y_prob=probs,
        class_names=class_names,
        threshold_target=0.80,
    )

    metrics["architecture"] = arch_name
    metrics["parameters"] = num_params
    metrics["training_time_sec"] = round(t_train, 3)
    metrics["inference_latency_ms_per_slice"] = round(inf_latency_ms, 3)

    return metrics


def run_foundational_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_names: List[str],
    backbones: Optional[List[str]] = None,
    epochs: int = 2,
    lr: float = 0.001,
    batch_size: int = 16,
    image_size: int = 224,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs foundational benchmark across all specified deep vision backbones."""
    if backbones is None:
        backbones = list(FOUNDATION_ARCH_MAP.keys())

    results = {}
    for name in backbones:
        cur_lr = lr if name != "convnext_small" else lr * 0.5
        res = train_and_eval_foundation_model(
            arch_name=name,
            train_df=train_df,
            test_df=test_df,
            class_names=class_names,
            epochs=epochs,
            lr=cur_lr,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
        )
        results[name] = res

    return results

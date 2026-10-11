"""Foundational Vision Model Training and Benchmark Engine for Diabetic Retinopathy.

Evaluates 4 Premier Vision Foundation Backbones:
1. DenseNet-121 (Radiology & Ophthalmology Dense Feature Reuse Standard)
2. ResNet-50 (Deep Residual Learning Foundation)
3. ConvNeXt-Small (Modern 2020s Pure Convolutional Foundation)
4. EfficientNet-B0 (Compound Scaled Inverted Bottleneck Foundation)

Operates on the harmonized dual database (EyePACS/APTOS + Messidor-2/IDRiD),
calculating full clinical metrics including Quadratic Weighted Kappa (QWK) and Referral Sensitivity.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.evaluator import evaluate_clinical_dr_metrics


class FundusDataset(Dataset):
    """PyTorch Dataset for retinal fundus scans with standard clinical preprocessing."""

    def __init__(self, samples: List[Any], transform: Optional[Any] = None, image_size: int = 224):
        self.samples = samples
        self.image_size = image_size
        self.transform = transform or T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        tensor = self.transform(img)
        label = int(s.label_idx)
        return tensor, label, s.image_id


def build_foundation_backbone(arch_name: str, num_classes: int = 5, dropout: float = 0.3) -> nn.Module:
    """Builds a vision foundation model with customized multi-class classification head."""
    if arch_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch_name == "convnext_small":
        model = models.convnext_small(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif arch_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(
            f"Unsupported foundation architecture: {arch_name}. "
            "Choose from ['densenet121', 'resnet50', 'convnext_small', 'efficientnet_b0']."
        )

    return model


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_foundation_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 3,
    lr: float = 0.0005,
    weight_decay: float = 0.0001,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Trains a foundation vision model using AdamW and Cosine Annealing schedule."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "epoch_times": []}
    t_start = time.perf_counter()

    for epoch in range(epochs):
        t_ep_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        total_batches = 0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        scheduler.step()
        ep_time = time.perf_counter() - t_ep_start
        mean_loss = running_loss / max(1, total_batches)
        history["train_loss"].append(mean_loss)
        history["epoch_times"].append(ep_time)

    total_train_time = time.perf_counter() - t_start
    return {
        "total_train_time_sec": round(total_train_time, 2),
        "history": history,
    }


def evaluate_foundation_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates the foundation vision model on held-out test split, computing clinical metrics."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(dev)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.tolist())

    total_inf_time = time.perf_counter() - t_inf_start
    total_samples = len(all_targets)
    inf_latency_ms = (total_inf_time / max(1, total_samples)) * 1000.0

    metrics = evaluate_clinical_dr_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_prob=np.array(all_probs),
        class_names=class_names,
        referral_threshold=2,
    )

    metrics["inference_latency_ms"] = round(inf_latency_ms, 3)
    metrics["total_inference_time_sec"] = round(total_inf_time, 3)
    metrics["total_evaluated_samples"] = total_samples

    return metrics


def run_foundational_benchmark(
    train_samples: List[Any],
    test_samples: List[Any],
    class_names: List[str],
    backbones: Optional[List[str]] = None,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 0.0005,
    device: str = "cpu",
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Executes the foundational vision model benchmark across all specified architectures."""
    if backbones is None:
        backbones = ["densenet121", "resnet50", "convnext_small", "efficientnet_b0"]

    train_ds = FundusDataset(train_samples, image_size=224)
    test_ds = FundusDataset(test_samples, image_size=224)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else Path("results/checkpoints")
    ckpt_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for arch in backbones:
        cur_lr = lr if arch != "convnext_small" else lr * 0.5
        model = build_foundation_backbone(arch, num_classes=len(class_names))
        param_count = count_parameters(model)

        # Train
        train_res = train_foundation_model(
            model=model,
            train_loader=train_loader,
            epochs=epochs,
            lr=cur_lr,
            device=device,
        )

        # Evaluate
        eval_res = evaluate_foundation_model(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
        )

        eval_res["architecture"] = arch
        eval_res["parameters"] = param_count
        eval_res["parameters_million"] = round(param_count / 1e6, 2)
        eval_res["training_time_sec"] = train_res["total_train_time_sec"]

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_path / f"foundation_{arch}.pth")

        results[arch] = eval_res

    return results

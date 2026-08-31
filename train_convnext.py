"""
Fine-Tuning Engine for ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small).
Includes Automatic Mixed Precision (AMP), AdamW with differential learning rates,
Cosine Annealing with Warmup, Label Smoothing, and Model Checkpointing.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from convnext_models import get_convnext_parameter_groups
from evaluate import compute_all_metrics


class ConvNeXtTrainer:
    """
    Manages fine-tuning, validation, evaluation, and checkpointing for ConvNeXt architectures.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: List[str],
        device: torch.device,
        output_dir: Path,
        backbone_lr: float = 2e-5,
        head_lr: float = 2e-4,
        weight_decay: float = 0.05,
        label_smoothing: float = 0.1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm

        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        param_groups = get_convnext_parameter_groups(
            model=self.model,
            backbone_lr=backbone_lr,
            head_lr=head_lr,
            weight_decay=weight_decay
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_macro_f1": []
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        total = 0
        all_preds, all_targets, all_probas = [], [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probas = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probas, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            total += images.size(0)

        eval_loss = total_loss / total
        return eval_loss, np.array(all_targets), np.array(all_preds), np.array(all_probas)

    def fit(
        self,
        epochs: int = 15,
        warmup_epochs: int = 3
    ) -> Dict[str, Any]:
        """
        Executes fine-tuning with Cosine Annealing learning rate schedule.
        """
        total_steps = epochs
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(1, total_steps - warmup_epochs)
                return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        best_val_f1 = 0.0
        best_checkpoint_path = self.output_dir / "best_convnext_model.pth"

        print(f"[ConvNeXt-Trainer] Starting Fine-Tuning for {epochs} epochs (Warmup: {warmup_epochs} epochs)...")
        print(f"[ConvNeXt-Trainer] Target Device: {self.device} (Mixed Precision AMP: {self.use_amp})")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, y_true_val, y_pred_val, y_proba_val = self.evaluate_loader(self.val_loader)

            val_metrics = compute_all_metrics(y_true_val, y_pred_val, y_proba_val, self.class_names)
            val_acc = val_metrics["Accuracy"]
            val_f1 = val_metrics["Macro_F1"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_f1)

            scheduler.step()

            # Checkpoint best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics
                }, best_checkpoint_path)
                saved_tag = " -> [SAVED BEST MODEL]"
            else:
                saved_tag = ""

            print(f" Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%{saved_tag}")

        elapsed = time.time() - start_time
        print(f"[ConvNeXt-Trainer] Fine-tuning completed in {elapsed/60:.2f} minutes. Best Val F1: {best_val_f1*100:.2f}%")

        # Load best checkpoint for test set evaluation
        print(f"[ConvNeXt-Trainer] Loading best checkpoint from {best_checkpoint_path} for test evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, y_true_test, y_pred_test, y_proba_test = self.evaluate_loader(self.test_loader)
        test_metrics = compute_all_metrics(y_true_test, y_pred_test, y_proba_test, self.class_names)

        # Plot training curves
        self.plot_training_curves(self.output_dir / "training_curves_convnext.png")

        return {
            "test_metrics": test_metrics,
            "y_true_test": y_true_test,
            "y_pred_test": y_pred_test,
            "y_proba_test": y_proba_test,
            "history": self.history
        }

    def plot_training_curves(self, output_path: Path):
        """
        Plots and saves training and validation loss/accuracy curves.
        """
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(12, 5), dpi=300)
        sns.set_theme(style="whitegrid")

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], 'o-', label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(epochs, self.history["val_loss"], 's--', label="Val Loss", color="#ff7f0e", lw=2)
        plt.title("ConvNeXt Loss vs. Epochs", fontsize=12, weight='bold')
        plt.xlabel("Epoch", fontsize=10, weight='bold')
        plt.ylabel("Loss", fontsize=10, weight='bold')
        plt.legend(frameon=True)

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, [a*100 for a in self.history["train_acc"]], 'o-', label="Train Acc", color="#2ca02c", lw=2)
        plt.plot(epochs, [a*100 for a in self.history["val_acc"]], 's--', label="Val Acc", color="#d62728", lw=2)
        plt.title("ConvNeXt Accuracy (%) vs. Epochs", fontsize=12, weight='bold')
        plt.xlabel("Epoch", fontsize=10, weight='bold')
        plt.ylabel("Accuracy (%)", fontsize=10, weight='bold')
        plt.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[ConvNeXt-Trainer] Training curves saved to: {output_path}")

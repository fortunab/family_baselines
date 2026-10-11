"""Federated Server implementation for Model Aggregation and Global Evaluation.

Coordinates communication rounds across clinical hospital nodes:
- Distributes consensus parameters
- Executes FedAvg / FedProx weighted parameter aggregation
- Evaluates global performance on held-out test cohort
- Identifies optimal clinical checkpoint
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.client import DermoscopyDataset, FederatedClient
from src.dataset_generator import CLASS_NAMES, MALIGNANT_CLASSES, DermoscopySample
from src.evaluator import evaluate_clinical_skin_lesions
from src.models import clone_state_dict, count_parameters, get_model

logger = logging.getLogger("FederatedServer")


class FederatedServer:
    """Central coordinator aggregating federated updates from clinical hospital clients."""

    def __init__(
        self,
        test_samples: List[DermoscopySample],
        model_arch: str = "resnet_derm",
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.25,
        device: str = "cpu",
        class_names: List[str] = None,
        malignant_classes: List[str] = None,
    ):
        self.device = torch.device(device)
        self.class_names = class_names or CLASS_NAMES
        self.malignant_classes = malignant_classes or MALIGNANT_CLASSES
        self.test_samples = test_samples

        # Global consensus model
        self.global_model = get_model(
            architecture=model_arch,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        param_count, param_mb = count_parameters(self.global_model)
        logger.info(f"Global Model initialized: {param_count:,} parameters ({param_mb:.2f} MB payload per round).")

        # Test evaluation loader
        self.eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        test_dataset = DermoscopyDataset(self.test_samples, transform=self.eval_transform)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=False,
        )

        self.criterion = nn.CrossEntropyLoss()

        # Tracking history
        self.history: Dict[str, Any] = {
            "rounds": [],
            "test_loss": [],
            "accuracy": [],
            "balanced_accuracy": [],
            "cohen_kappa": [],
            "macro_f1": [],
            "referral_sensitivity": [],
            "referral_specificity": [],
            "client_train_losses": {},
            "client_divergences": {},
        }
        self.best_kappa = -1.0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Returns deep cloned copy of current global parameters."""
        return clone_state_dict(self.global_model.state_dict())

    def aggregate_weights(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """FedAvg / FedProx Parameter Aggregation weighted by client sample volume:

        w_{global} = sum_k (n_k / N) * w_k
        """
        total_samples = sum(u["sample_count"] for u in client_updates)
        if total_samples == 0:
            return self.get_global_weights()

        aggregated_state: Dict[str, torch.Tensor] = {}
        first_client_dict = client_updates[0]["updated_state_dict"]

        for key in first_client_dict:
            # Check if floating point parameter
            if first_client_dict[key].is_floating_point():
                weighted_param = torch.zeros_like(first_client_dict[key], device=self.device)
                for update in client_updates:
                    w_k = update["sample_count"] / total_samples
                    client_param = update["updated_state_dict"][key].to(self.device)
                    weighted_param += w_k * client_param
                aggregated_state[key] = weighted_param
            else:
                # Long or integer buffers (e.g. num_batches_tracked)
                aggregated_state[key] = copy.deepcopy(first_client_dict[key])

        self.global_model.load_state_dict(aggregated_state)
        return aggregated_state

    def evaluate_global_model(self) -> Tuple[float, Dict[str, Any]]:
        """Evaluates current global consensus model on held-out test cohort."""
        self.global_model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.global_model(images)
                loss = self.criterion(logits, targets)
                total_loss += loss.item() * len(targets)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(1, len(self.test_samples))
        y_prob = np.concatenate(all_probs, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        metrics = evaluate_clinical_skin_lesions(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=self.class_names,
            malignant_classes=self.malignant_classes,
        )
        metrics["test_loss"] = round(float(avg_loss), 4)

        return avg_loss, metrics

    def run_communication_round(
        self,
        round_idx: int,
        clients: List[FederatedClient],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes one complete global federated communication round."""
        global_weights = self.get_global_weights()

        client_updates = []
        round_client_metrics = {}

        # 1. Distribute global weights and execute local training
        for client in clients:
            client.receive_global_weights(global_weights)
            update_data = client.local_train()
            client_updates.append(update_data)

            c_id = client.client_id
            if c_id not in self.history["client_train_losses"]:
                self.history["client_train_losses"][c_id] = []
                self.history["client_divergences"][c_id] = []
            self.history["client_train_losses"][c_id].append(update_data["train_loss"])
            self.history["client_divergences"][c_id].append(update_data["divergence"])

            round_client_metrics[c_id] = {
                "train_loss": update_data["train_loss"],
                "divergence": update_data["divergence"],
                "sample_count": update_data["sample_count"],
            }

        # 2. Server aggregates weights
        self.aggregate_weights(client_updates)

        # 3. Global model evaluation
        test_loss, global_metrics = self.evaluate_global_model()

        # Update history
        self.history["rounds"].append(round_idx)
        self.history["test_loss"].append(global_metrics["test_loss"])
        self.history["accuracy"].append(global_metrics["accuracy"])
        self.history["balanced_accuracy"].append(global_metrics["balanced_accuracy"])
        self.history["cohen_kappa"].append(global_metrics["cohen_kappa"])
        self.history["macro_f1"].append(global_metrics["macro_f1"])

        ref_dict = global_metrics["binary_malignancy_referral"]
        self.history["referral_sensitivity"].append(ref_dict["referral_sensitivity"])
        self.history["referral_specificity"].append(ref_dict["referral_specificity"])

        # Checkpoint if best kappa
        if global_metrics["cohen_kappa"] > self.best_kappa:
            self.best_kappa = global_metrics["cohen_kappa"]
            self.best_state_dict = clone_state_dict(self.global_model.state_dict())

        return global_metrics, round_client_metrics

    def save_checkpoint(self, output_path: str):
        """Saves best model weights to disk."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state_to_save = self.best_state_dict or self.global_model.state_dict()
        torch.save(
            {
                "best_kappa": self.best_kappa,
                "model_state_dict": state_to_save,
                "history": self.history,
            },
            p,
        )
        logger.info(f"Saved federated model checkpoint to {p}")

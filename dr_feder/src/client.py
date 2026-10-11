"""Federated Client for Local Hospital Retinal Fundus Training.

Simulates a private clinical ophthalmology center (e.g. EyePACS screening clinic,
Paris hospital, IDRiD tertiary institute) optimizing exclusively on local patient fundus scans.
Supports:
- Local AdamW / SGD training loops
- FedProx proximal regularization against Non-IID label skew
- Local weight divergence measurement
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset_generator import RetinalFundusSample
from src.models import (
    clone_state_dict,
    compute_proximal_term,
    compute_weight_divergence,
    get_model,
)


class RetinalFundusDataset(Dataset):
    """PyTorch Dataset for retinal fundus photographs."""

    def __init__(self, samples: List[RetinalFundusSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        return img_tensor, s.grade


class FederatedClient:
    """Represents a private clinical ophthalmology center in the federated network."""

    def __init__(
        self,
        client_id: int,
        client_name: str,
        samples: List[RetinalFundusSample],
        model_arch: str = "resnet_retina",
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.25,
        lr: float = 0.002,
        weight_decay: float = 0.0001,
        batch_size: int = 16,
        local_epochs: int = 2,
        algorithm: str = "fedprox",
        fedprox_mu: float = 0.01,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.samples = samples
        self.algorithm = algorithm
        self.fedprox_mu = fedprox_mu
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        # Local model
        self.model = get_model(
            architecture=model_arch,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        # Global consensus reference for FedProx
        self.global_reference = get_model(
            architecture=model_arch,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        # Ophthalmic augmentations
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        dataset = RetinalFundusDataset(self.samples, transform=self.train_transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(self.samples))),
            shuffle=True,
            drop_last=False,
        )

        self.criterion = nn.CrossEntropyLoss()

    def receive_global_weights(self, global_state_dict: Dict[str, torch.Tensor]):
        """Sets client model weights to the received global consensus state."""
        self.model.load_state_dict(global_state_dict)
        self.global_reference.load_state_dict(global_state_dict)
        for param in self.global_reference.parameters():
            param.requires_grad = False

    def local_train(self) -> Dict[str, Any]:
        """Performs local client training epochs on private patient fundus dataset."""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_loss = 0.0
        total_batches = 0

        initial_state = clone_state_dict(self.model.state_dict())

        for epoch in range(self.local_epochs):
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                # FedProx proximal penalty
                if self.algorithm == "fedprox" and self.fedprox_mu > 0.0:
                    prox_term = compute_proximal_term(self.model, self.global_reference)
                    loss += self.fedprox_mu * prox_term

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)

        # Compute weight divergence
        final_state = self.model.state_dict()
        divergence = compute_weight_divergence(final_state, initial_state)

        return {
            "client_id": self.client_id,
            "client_name": self.client_name,
            "sample_count": len(self.samples),
            "train_loss": round(float(avg_loss), 4),
            "divergence": round(float(divergence), 4),
            "updated_state_dict": clone_state_dict(final_state),
        }

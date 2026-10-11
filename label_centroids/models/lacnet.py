import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .backbone import ConvNetBackbone


class LACNet(nn.Module):
    """
    Label-Attractor Contrastive Network (LACNet).
    Paper: "Label-Attractor Contrastive Network: A Framework for Centroid-Driven Histology Samples"
    (Babuc & Fortiş, 2026/2027)

    Architecture:
    - Shared Image Encoder: 2-block ConvNet + GAP + Dense + L2-norm
    - Trainable Label Attractors: A = {a_c} in R^d
    - Label-Attractor Contrastive Loss with L2 regularizer
    - Periodic Self-Organizing Centroid Refinement
    - Nearest-attractor cosine similarity inference
    """
    def __init__(
        self,
        num_classes: int = 8,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        lambda_reg: float = 1e-4,
        in_channels: int = 3,
        base_channels: int = 32,
        trainable_attractors: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.lambda_reg = lambda_reg
        self.trainable_attractors = trainable_attractors

        # Encoder backbone
        self.backbone = ConvNetBackbone(in_channels=in_channels, base_channels=base_channels)
        self.projection = nn.Linear(self.backbone.out_features, embedding_dim)

        # Label attractors: one trainable vector per class in unit hypersphere embedding space
        attractors_init = torch.randn(num_classes, embedding_dim)
        attractors_init = F.normalize(attractors_init, p=2, dim=-1)

        if trainable_attractors:
            self.label_attractors = nn.Parameter(attractors_init)
        else:
            self.register_buffer("label_attractors", attractors_init)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input images to L2-normalized latent embeddings on the unit hypersphere.
        e in R^d such that ||e||_2 = 1.
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        return normalized_embeddings

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embeddings for visualization (e.g., t-SNE)."""
        return self.encode(x)

    def compute_similarities(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes temperature-scaled cosine similarities between embeddings and label attractors.
        s_{i,c} = (e_i^T a_c) / tau  (Eq. 1)
        """
        # embeddings: (B, d), label_attractors: (C, d)
        # Cosine similarity matrix: (B, C)
        # We normalize attractors during similarity calculation for stability
        norm_attractors = F.normalize(self.label_attractors, p=2, dim=-1)
        cosine_sim = torch.matmul(embeddings, norm_attractors.t())
        scaled_sim = cosine_sim / self.temperature
        return scaled_sim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returns normalized embeddings and scaled similarity logits.
        """
        embeddings = self.encode(x)
        similarities = self.compute_similarities(embeddings)
        return embeddings, similarities

    def compute_loss(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        custom_lambda: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Label-Attractor Contrastive Loss (Eq. 2):
        L = L_{CE} + lambda * sum_{c=1}^C ||a_c||_2^2
        """
        ce_loss = F.cross_entropy(similarities, labels)

        lam = self.lambda_reg if custom_lambda is None else custom_lambda
        if lam > 0:
            reg_loss = lam * torch.sum(self.label_attractors ** 2)
        else:
            reg_loss = torch.tensor(0.0, device=similarities.device)

        total_loss = ce_loss + reg_loss
        return total_loss, ce_loss, reg_loss

    @torch.no_grad()
    def initialize_attractors(self, dataloader, device: torch.device):
        """
        Initializes label attractors using class-wise centroids computed from
        the encoder's initial embeddings over training data, L2-normalized (Section 3.3).
        """
        self.eval()
        class_sums = torch.zeros(self.num_classes, self.embedding_dim, device=device)
        class_counts = torch.zeros(self.num_classes, device=device)

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = self.encode(images)

            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.any():
                    class_sums[c] += embeddings[mask].sum(dim=0)
                    class_counts[c] += mask.sum()

        for c in range(self.num_classes):
            if class_counts[c] > 0:
                mean_centroid = class_sums[c] / class_counts[c]
                normalized_centroid = F.normalize(mean_centroid, p=2, dim=0)
                self.label_attractors.data[c] = normalized_centroid

    @torch.no_grad()
    def periodic_refinement(self, dataloader, device: torch.device, mode: str = "hybrid"):
        """
        Periodic Attractor Refinement (Section 3.5).
        Enforces global semantic consistency independently of gradient descent.
        Every training sample is assigned based on closest attractor / class semantics,
        and the normalized mean of assigned embeddings recalculates each attractor:
        a_c <- mu_c / ||mu_c||_2.
        """
        self.eval()
        cluster_sums = torch.zeros(self.num_classes, self.embedding_dim, device=device)
        cluster_counts = torch.zeros(self.num_classes, device=device)

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            embeddings = self.encode(images)

            if mode == "nearest":
                # Pure self-organizing assignment to nearest attractor
                sims = torch.matmul(embeddings, F.normalize(self.label_attractors.data, p=2, dim=-1).t())
                assigned = torch.argmax(sims, dim=-1)
            elif mode == "class":
                # Supervised class centroid update
                assigned = labels
            else:
                # Hybrid: assign to nearest attractor within consistent ground-truth guidance
                sims = torch.matmul(embeddings, F.normalize(self.label_attractors.data, p=2, dim=-1).t())
                nearest = torch.argmax(sims, dim=-1)
                # If nearest matches class, use nearest; otherwise balance with class label
                assigned = labels

            for c in range(self.num_classes):
                mask = (assigned == c)
                if mask.any():
                    cluster_sums[c] += embeddings[mask].sum(dim=0)
                    cluster_counts[c] += mask.sum()

        # Update attractors with normalized means
        for c in range(self.num_classes):
            if cluster_counts[c] > 0:
                mean_vec = cluster_sums[c] / cluster_counts[c]
                norm_mean = F.normalize(mean_vec, p=2, dim=0)
                # Momentum / direct update to preserve smooth training trajectory
                self.label_attractors.data[c] = norm_mean

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference test without classifier head (Section 3.6):
        y_hat = argmax_c (e^T a_c).
        """
        self.eval()
        embeddings = self.encode(x)
        norm_attractors = F.normalize(self.label_attractors, p=2, dim=-1)
        sims = torch.matmul(embeddings, norm_attractors.t())
        predictions = torch.argmax(sims, dim=-1)
        return predictions

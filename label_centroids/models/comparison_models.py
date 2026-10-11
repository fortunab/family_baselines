import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .backbone import ConvNetBackbone


class TripletLossCNN(nn.Module):
    """
    Metric learning baseline with Triplet Loss (Table 1).
    Uses the 2-block ConvNet backbone, embeds samples,
    and trains with online hard triplet mining or batch semi-hard triplet loss.
    Classification is performed via 1-NN or nearest-class-mean.
    """
    def __init__(self, num_classes: int = 8, embedding_dim: int = 128, margin: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.backbone = ConvNetBackbone()
        self.projection = nn.Linear(self.backbone.out_features, embedding_dim)
        self.register_buffer("class_prototypes", torch.zeros(num_classes, embedding_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.projection(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def update_prototypes(self, dataloader, device):
        self.eval()
        sums = torch.zeros(self.num_classes, self.embedding_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                embs = self.encode(imgs)
                for c in range(self.num_classes):
                    m = (lbls == c)
                    if m.any():
                        sums[c] += embs[m].sum(dim=0)
                        counts[c] += m.sum()
        for c in range(self.num_classes):
            if counts[c] > 0:
                self.class_prototypes[c] = F.normalize(sums[c] / counts[c], p=2, dim=0)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.encode(x)
        # Cosine similarity to class prototypes
        sims = torch.matmul(embs, self.class_prototypes.t())
        return torch.argmax(sims, dim=-1)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network baseline (Snell et al. 2017, Table 1).
    Computes class prototypes as the mean support embeddings per episode/batch.
    Classification is performed via Euclidean distance / cosine distance to prototypes.
    """
    def __init__(self, num_classes: int = 8, embedding_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.backbone = ConvNetBackbone()
        self.projection = nn.Linear(self.backbone.out_features, embedding_dim)
        self.register_buffer("running_prototypes", torch.zeros(num_classes, embedding_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.projection(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def compute_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute episodic batch prototypes
        prototypes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                proto = embeddings[mask].mean(dim=0)
            else:
                proto = self.running_prototypes[c]
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)
        prototypes = F.normalize(prototypes, p=2, dim=-1)

        # Update running prototypes
        with torch.no_grad():
            self.running_prototypes.copy_(0.9 * self.running_prototypes + 0.1 * prototypes)

        # Compute distances / similarities
        sims = torch.matmul(embeddings, prototypes.t()) / 0.1
        return F.cross_entropy(sims, labels)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.encode(x)
        sims = torch.matmul(embs, F.normalize(self.running_prototypes, p=2, dim=-1).t())
        return torch.argmax(sims, dim=-1)


class SupConCNN(nn.Module):
    """
    Supervised Contrastive Learning baseline (Khosla et al. 2020 / Omeroglu 2024, Table 1).
    Encourages all samples from the same class to cluster together while pushing
    different classes apart using batch-level sample comparisons.
    """
    def __init__(self, num_classes: int = 8, embedding_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.backbone = ConvNetBackbone()
        self.projection = nn.Linear(self.backbone.out_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.projection(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode(x)
        logits = self.classifier(emb)
        return emb, logits


class MultiPrototypeCNN(nn.Module):
    """
    Multi-prototype CNN baseline (Deuschel et al. 2021, Table 1).
    Maintains multiple (K=2) sub-prototypes per class to capture multi-modal distributions.
    """
    def __init__(self, num_classes: int = 8, num_prototypes_per_class: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.embedding_dim = embedding_dim
        self.backbone = ConvNetBackbone()
        self.projection = nn.Linear(self.backbone.out_features, embedding_dim)

        # K prototypes per class: shape (C, K, d)
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(num_classes, num_prototypes_per_class, embedding_dim), p=2, dim=-1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.projection(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = self.encode(x)  # (B, d)
        norm_protos = F.normalize(self.prototypes, p=2, dim=-1)  # (C, K, d)
        # Compute similarity to all (C * K) prototypes
        # (B, 1, 1, d) * (1, C, K, d) -> sum over d -> (B, C, K)
        b = embs.shape[0]
        sims = torch.einsum("bd,ckd->bck", embs, norm_protos)
        # Class score is maximum similarity over the class sub-prototypes (or logsumexp)
        class_scores, _ = torch.max(sims, dim=-1)  # (B, C)
        return class_scores / 0.1

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.forward(x)
        return torch.argmax(scores, dim=-1)

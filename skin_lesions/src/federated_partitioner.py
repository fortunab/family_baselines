"""Non-IID Dirichlet Federated Partitioner for Multi-Hospital Dermatology Data.

Allocates patient dermoscopy cases across clinical hospital nodes according to
a Dirichlet distribution Dir(alpha) to simulate real-world clinical label skew.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from src.dataset_generator import CLASS_NAMES, DermoscopySample

logger = logging.getLogger("FederatedPartitioner")


@dataclass
class ClientPartition:
    client_id: int
    client_name: str
    sample_indices: List[int]
    sample_count: int
    class_counts: Dict[str, int]
    malignant_count: int
    benign_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def partition_dataset_dirichlet(
    samples: List[DermoscopySample],
    num_clients: int = 4,
    client_names: List[str] = None,
    alpha: float = 0.5,
    seed: int = 42,
) -> Tuple[Dict[int, ClientPartition], np.ndarray]:
    """Partitions training samples across clients using Dirichlet distribution Dir(alpha).

    Returns:
        client_partitions: Dict mapping client_id -> ClientPartition
        client_label_matrix: 2D array of shape (num_clients, num_classes) with sample counts
    """
    if client_names is None or len(client_names) != num_clients:
        client_names = [f"Hospital_{i+1:02d}" for i in range(num_clients)]

    train_indices = [i for i, s in enumerate(samples) if s.split == "train"]
    y_train = np.array([samples[i].class_idx for i in train_indices])

    num_classes = len(CLASS_NAMES)
    client_sample_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    rng = np.random.default_rng(seed)

    # For each class, sample client proportions from Dir(alpha)
    for c in range(num_classes):
        idx_c = np.where(y_train == c)[0]
        rng.shuffle(idx_c)

        # Dirichlet proportions for class c across all clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Ensure at least 1 sample if available
        counts = (proportions * len(idx_c)).astype(int)

        # Adjust rounding errors
        diff = len(idx_c) - counts.sum()
        for d in range(abs(diff)):
            if diff > 0:
                counts[d % num_clients] += 1
            elif diff < 0:
                max_c = np.argmax(counts)
                if counts[max_c] > 0:
                    counts[max_c] -= 1

        # Distribute original sample indices
        start_idx = 0
        for client_id in range(num_clients):
            n_assign = counts[client_id]
            assigned_c_indices = idx_c[start_idx : start_idx + n_assign]
            # Map back to original dataset index in samples list
            for local_idx in assigned_c_indices:
                client_sample_indices[client_id].append(train_indices[local_idx])
            start_idx += n_assign

    # Build structured client partition metadata
    client_partitions: Dict[int, ClientPartition] = {}
    client_label_matrix = np.zeros((num_clients, num_classes), dtype=int)

    for client_id in range(num_clients):
        s_indices = client_sample_indices[client_id]
        cls_counts = {c_name: 0 for c_name in CLASS_NAMES}
        mal_count = 0
        ben_count = 0

        for idx in s_indices:
            s_obj = samples[idx]
            cls_counts[s_obj.class_name] += 1
            client_label_matrix[client_id, s_obj.class_idx] += 1
            if s_obj.is_malignant:
                mal_count += 1
            else:
                ben_count += 1

        part = ClientPartition(
            client_id=client_id,
            client_name=client_names[client_id],
            sample_indices=s_indices,
            sample_count=len(s_indices),
            class_counts=cls_counts,
            malignant_count=mal_count,
            benign_count=ben_count,
        )
        client_partitions[client_id] = part

    return client_partitions, client_label_matrix


def save_partition_manifest(
    client_partitions: Dict[int, ClientPartition],
    output_path: Union[str, Path] = "data/manifests/federated_partitions.json",
):
    """Saves partition information to JSON manifest."""
    p_path = Path(output_path)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    export_dict = {
        "num_clients": len(client_partitions),
        "clients": {str(cid): p.to_dict() for cid, p in client_partitions.items()},
    }
    with open(p_path, "w", encoding="utf-8") as f:
        json.dump(export_dict, f, indent=2)
    logger.info(f"Saved federated partition manifest to {p_path}")

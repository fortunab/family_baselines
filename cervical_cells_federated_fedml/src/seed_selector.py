"""
Dynamic High-Entropy Random Seed Generator & "Random Best Select" Optimizer.
Strictly eliminates hardcoded seed 42 in favor of cryptographic entropy
and clinical partition balance optimization across decentralized FedML nodes.
"""

import math
import random
import secrets
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch


def generate_dynamic_seed() -> int:
    """
    Generates a cryptographically secure 6-digit random seed in [100000, 999999].
    Guarantees seed 42 is NEVER produced.
    """
    while True:
        # Cryptographic high-entropy seed
        seed = secrets.randbelow(900000) + 100000
        if seed != 42:
            return seed


def apply_random_seed(seed: int) -> int:
    """
    Applies the active random seed across Python, NumPy, and PyTorch backends.
    Rejects seed 42 with an explicit notice.
    """
    if seed == 42:
        print("[Seed-Security] Warning: Seed 42 is disabled per benchmark requirements.")
        print("[Seed-Security] Replacing seed 42 with dynamic cryptographic seed...")
        seed = generate_dynamic_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def compute_distribution_entropy(counts: Counter, num_classes: int = 7) -> float:
    """Computes normalized Shannon entropy for class distribution in a node."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for cls_idx in range(num_classes):
        cnt = counts.get(cls_idx, 0)
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log2(p)

    max_entropy = math.log2(num_classes)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def evaluate_partition_balance(
    client_dfs: list[pd.DataFrame],
    num_classes: int = 7,
) -> float:
    """
    Calculates the mean normalized Shannon class distribution entropy across all FedML hospital silos.
    Higher entropy indicates a more balanced representation across decentralized clients.
    """
    if not client_dfs:
        return 0.0

    entropies = []
    for client_df in client_dfs:
        if "label" in client_df.columns:
            counts = Counter(client_df["label"].tolist())
            entropies.append(compute_distribution_entropy(counts, num_classes))

    return float(np.mean(entropies)) if entropies else 0.0


def find_best_partition_seed(
    partition_fn: Callable[[int], tuple[list[pd.DataFrame], pd.DataFrame]],
    num_candidates: int = 5,
    num_classes: int = 7,
) -> tuple[int, list[pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    """
    "Random Best Select" Optimizer:
    Evaluates K candidate high-entropy random seeds (excluding 42) and picks the one
    yielding the highest partition balance (mean Shannon entropy across clients).
    """
    print("\n" + "=" * 80)
    print("  FEDML RANDOM BEST SELECT: MULTI-CANDIDATE SEED ENTROPY SEARCH")
    print("=" * 80)
    print(f"[*] Evaluating K={num_candidates} cryptographic seed candidates...")

    best_seed: int | None = None
    best_entropy: float = -1.0
    best_client_dfs: list[pd.DataFrame] | None = None
    best_test_df: pd.DataFrame | None = None
    candidate_records: list[dict[str, Any]] = []

    for idx in range(num_candidates):
        candidate_seed = generate_dynamic_seed()
        client_dfs, test_df = partition_fn(candidate_seed)
        entropy = evaluate_partition_balance(client_dfs, num_classes=num_classes)

        total_train_samples = sum(len(df) for df in client_dfs)
        candidate_records.append(
            {
                "candidate_index": idx + 1,
                "seed": candidate_seed,
                "mean_client_entropy": round(entropy, 4),
                "num_clients": len(client_dfs),
                "total_train_samples": total_train_samples,
                "test_samples": len(test_df),
            }
        )

        print(
            f"  Candidate {idx + 1:02d}/{num_candidates:02d} | "
            f"Seed: {candidate_seed} | Mean Shannon Entropy: {entropy:.4f} | "
            f"Train/Test: {total_train_samples}/{len(test_df)}"
        )

        if entropy > best_entropy:
            best_entropy = entropy
            best_seed = candidate_seed
            best_client_dfs = client_dfs
            best_test_df = test_df

    assert best_seed is not None and best_client_dfs is not None and best_test_df is not None

    print("-" * 80)
    print(f"[BEST] Selected Optimal Seed: {best_seed} " f"(Peak Mean Entropy: {best_entropy:.4f})")
    print("=" * 80 + "\n")

    summary_info = {
        "best_seed": best_seed,
        "best_entropy": best_entropy,
        "num_candidates_evaluated": num_candidates,
        "candidate_records": candidate_records,
    }

    apply_random_seed(best_seed)
    return best_seed, best_client_dfs, best_test_df, summary_info

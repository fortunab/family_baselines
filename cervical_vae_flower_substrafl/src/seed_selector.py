"""
Cryptographic Random Seed Management and 'Random Best Select' Multi-Candidate Optimizer.
Strictly prohibits seed 42 in favor of dynamic entropy optimization.
"""

import os
import random
import secrets
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

FORBIDDEN_SEEDS = {42}


def generate_dynamic_seed() -> int:
    """Generates a cryptographically strong 6-digit random integer seed."""
    while True:
        candidate = secrets.randbelow(900000) + 100000
        if candidate not in FORBIDDEN_SEEDS:
            return candidate


def apply_random_seed(seed: int | None = None) -> int:
    """
    Applies the specified or generated seed across all frameworks.
    Strictly intercepts and replaces forbidden seeds (such as seed 42).
    """
    if seed is None or seed == 0 or seed in FORBIDDEN_SEEDS:
        if seed in FORBIDDEN_SEEDS:
            print(f"[POLICY WARNING] Seed {seed} is prohibited by experimental policy. Substituting dynamic seed.")
        seed = generate_dynamic_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def compute_partition_shannon_entropy(df: pd.DataFrame, num_classes: int = 7) -> float:
    """Computes normalized Shannon entropy for label distribution within a partition."""
    if len(df) == 0:
        return 0.0

    counts = df["label"].value_counts().values
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]

    raw_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_classes) if num_classes > 1 else 1.0

    return float(raw_entropy / max_entropy)


def find_best_partition_seed(
    partition_fn: Callable[[int], tuple[list[pd.DataFrame], pd.DataFrame]],
    num_candidates: int = 5,
    num_classes: int = 7,
) -> tuple[int, list[pd.DataFrame], pd.DataFrame, list[dict[str, Any]]]:
    """
    Evaluates K candidate cryptographic seeds and selects the seed maximizing
    mean Shannon class distribution entropy across all decentralized hospital silos.
    """
    print("\n" + "=" * 80)
    print("  FLOWER / SUBSTRA RANDOM BEST SELECT: MULTI-CANDIDATE SEED ENTROPY SEARCH")
    print("=" * 80)
    print(f"[*] Evaluating K={num_candidates} cryptographic seed candidates...")

    best_seed: int | None = None
    best_mean_entropy = -1.0
    best_partitions: tuple[list[pd.DataFrame], pd.DataFrame] | None = None
    candidate_summaries: list[dict[str, Any]] = []

    for i in range(num_candidates):
        cand_seed = generate_dynamic_seed()
        client_dfs, test_df = partition_fn(cand_seed)

        entropies = [compute_partition_shannon_entropy(c_df, num_classes) for c_df in client_dfs]
        mean_ent = float(np.mean(entropies))

        total_train = sum(len(c_df) for c_df in client_dfs)
        total_test = len(test_df)

        candidate_summaries.append(
            {
                "candidate_idx": i + 1,
                "seed": cand_seed,
                "mean_entropy": mean_ent,
                "entropies": entropies,
                "train_samples": total_train,
                "test_samples": total_test,
            }
        )

        print(
            f"  Candidate {i + 1:02d}/{num_candidates:02d} | Seed: {cand_seed} | "
            f"Mean Shannon Entropy: {mean_ent:.4f} | Train/Test: {total_train}/{total_test}"
        )

        if mean_ent > best_mean_entropy:
            best_mean_entropy = mean_ent
            best_seed = cand_seed
            best_partitions = (client_dfs, test_df)

    assert best_seed is not None and best_partitions is not None
    apply_random_seed(best_seed)

    print("-" * 80)
    print(f"[BEST] Selected Optimal Seed: {best_seed} (Peak Mean Entropy: {best_mean_entropy:.4f})")
    print("=" * 80 + "\n")

    return best_seed, best_partitions[0], best_partitions[1], candidate_summaries

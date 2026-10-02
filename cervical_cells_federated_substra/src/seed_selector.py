"""
Dynamic High-Entropy Random Seed Generator & "Random Best Select" Optimizer.
Strictly eliminates hardcoded seed 42 in favor of cryptographic entropy
and clinical partition balance optimization across decentralized Substra nodes.
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

    max_entropy = math.log2(num_classes) if num_classes > 1 else 1.0
    return entropy / max_entropy


def evaluate_partition_quality(
    client_dfs: list[pd.DataFrame],
    test_df: pd.DataFrame,
    num_classes: int = 7,
) -> float:
    """
    Evaluates partition quality across Substra nodes:
    - Higher average class entropy across nodes -> better clinical diversity
    - Lower size variance between nodes -> balanced federated convergence
    - Complete representation in holdout test set -> reliable centralized benchmark
    """
    if not client_dfs or len(test_df) == 0:
        return 0.0

    entropies = []
    sizes = []
    for c_df in client_dfs:
        sizes.append(len(c_df))
        if "label" in c_df.columns:
            counts = Counter(c_df["label"])
            # Map string labels to numeric indices if needed
            numeric_counts = Counter({i: v for i, (k, v) in enumerate(counts.items())})
            entropies.append(compute_distribution_entropy(numeric_counts, num_classes))
        else:
            entropies.append(0.5)

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    size_std = float(np.std(sizes)) if len(sizes) > 1 else 0.0
    size_penalty = min(0.5, size_std / (np.mean(sizes) + 1e-6))

    # Test set representation coverage
    test_classes = test_df["label"].nunique() if "label" in test_df.columns else num_classes
    test_coverage = test_classes / float(num_classes)

    score = (0.5 * avg_entropy) + (0.3 * test_coverage) - (0.2 * size_penalty)
    return max(0.0, score)


def select_best_random_seed(
    partition_fn: Callable[[int], tuple[list[pd.DataFrame], pd.DataFrame]],
    num_candidates: int = 5,
    verbose: bool = True,
) -> tuple[int, dict[str, Any]]:
    """
    "Random Best Select" Optimizer:
    Evaluates K candidate high-entropy random seeds, scores their partition distributions,
    and returns the highest-scoring seed for optimal federated stability.
    Strictly excludes seed 42.
    """
    candidates = []
    while len(candidates) < num_candidates:
        cand = generate_dynamic_seed()
        if cand not in candidates and cand != 42:
            candidates.append(cand)

    if verbose:
        print("\n" + "=" * 70)
        print("    SUBSTRA 'RANDOM BEST SELECT' DYNAMIC SEED OPTIMIZER (NO SEED 42)")
        print("=" * 70)
        print(f"  Evaluating {num_candidates} cryptographic candidate seeds for optimal distribution...")

    leaderboard = []
    for cand in candidates:
        client_dfs, test_df = partition_fn(cand)
        quality_score = evaluate_partition_quality(client_dfs, test_df)
        leaderboard.append(
            {
                "seed": cand,
                "score": quality_score,
                "test_samples": len(test_df),
                "client_samples": [len(c) for c in client_dfs],
            }
        )

    # Sort descending by quality score
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    best_entry = leaderboard[0]
    best_seed = int(best_entry["seed"])

    if verbose:
        print("\n  Candidate Seed Evaluation Leaderboard:")
        for rank, entry in enumerate(leaderboard, 1):
            star = " * [BEST SELECTED]" if rank == 1 else ""
            print(
                f"   #{rank} Seed: {entry['seed']} | Quality Score: {entry['score']:.4f} | "
                f"Test Set: {entry['test_samples']} | Clients: {entry['client_samples']}{star}"
            )
        print("=" * 70 + "\n")

    apply_random_seed(best_seed)
    return best_seed, best_entry

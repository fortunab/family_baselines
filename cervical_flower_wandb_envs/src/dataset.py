"""
Cervical Cytology (Herlev Pap Smear) 7-Class Dataset Pipeline & Non-IID Dirichlet Partitioning.
Partitions cytology data across K clinic clients while reserving a 15% holdout test set for the server.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "01_normal_superficiel",
    "02_normal_intermediate",
    "03_normal_columnar",
    "04_light_dysplastic",
    "05_moderate_dysplastic",
    "06_severe_dysplastic",
    "07_carcinoma_in_situ",
]

CLASS_SHORT_NAMES = [
    "SUPERFICIAL",
    "INTERMEDIATE",
    "COLUMNAR",
    "LIGHT_DYS",
    "MOD_DYS",
    "SEV_DYS",
    "CARCINOMA",
]

CLASS_KEYWORDS = [
    ["superfici"],
    ["intermediate"],
    ["columnar"],
    ["light", "mild"],
    ["moderate"],
    ["severe"],
    ["carcinoma", "situ"],
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}


def generate_synthetic_cervical_cells(
    output_dir: Path, samples_per_class: int = 15
) -> pd.DataFrame:
    """
    Fallback generator for synthetic cervical cytology cells if dataset is absent.
    Simulates pap smear single cell morphology with cytoplasm and nucleus.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for c_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            img = Image.new(
                "RGB",
                (160, 160),
                color=(
                    random.randint(230, 245),
                    random.randint(230, 245),
                    random.randint(235, 250),
                ),
            )
            draw = ImageDraw.Draw(img)

            # Draw cytoplasm (larger irregular oval)
            cyto_color = (
                random.randint(180, 210),
                random.randint(190, 225),
                random.randint(205, 235),
            )
            draw.ellipse(
                [(25, 20), (135, 140)],
                fill=cyto_color,
                outline=(150, 160, 180),
                width=1,
            )

            # Draw nucleus (dark violet/purple, larger in high grade dysplastic/cancer cells)
            n_ratio = 12 + c_idx * 4
            n_x = 80 + random.randint(-8, 8)
            n_y = 80 + random.randint(-8, 8)
            draw.ellipse(
                [(n_x - n_ratio, n_y - n_ratio), (n_x + n_ratio, n_y + n_ratio)],
                fill=(
                    random.randint(40, 75),
                    random.randint(20, 50),
                    random.randint(70, 115),
                ),
            )

            file_path = cls_dir / f"cell_{i:04d}.bmp"
            img.save(file_path)
            records.append(
                {
                    "filepath": str(file_path.resolve()),
                    "label": cls_name,
                    "label_idx": c_idx,
                    "class_short": CLASS_SHORT_NAMES[c_idx],
                }
            )

    return pd.DataFrame(records)


def discover_cervical_dataset(data_path: str) -> pd.DataFrame:
    """
    Discovers all cervical cytology cell images across the 7 classes,
    strictly excluding mask files (-d.bmp, -cyt.bmp).
    """
    candidate_roots = [
        Path(data_path),
        Path("C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/data"),
        Path("C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/data"),
        Path("C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_pathology_foundation_fastai/data"),
        Path("C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_pathology_foundation_skorch/data"),
        Path("../herlev_cervical_baseline/data"),
        Path("../herlev_original_mde_baseline/data"),
    ]

    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue

        records = []
        subdirs = [d for d in root.iterdir() if d.is_dir()]

        for c_idx, keywords in enumerate(CLASS_KEYWORDS):
            matching_dir = None
            for d in subdirs:
                d_lower = d.name.lower()
                if any(kw in d_lower for kw in keywords):
                    matching_dir = d
                    break

            if matching_dir is None:
                continue

            for img_path in matching_dir.glob("*.*"):
                name_lower = img_path.name.lower()
                # Strict exclusion of nucleus and cytoplasm mask files
                if "-d.bmp" in name_lower or "-cyt.bmp" in name_lower:
                    continue
                if name_lower.endswith((".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")):
                    records.append(
                        {
                            "filepath": str(img_path.resolve()),
                            "label": CLASS_NAMES[c_idx],
                            "label_idx": c_idx,
                            "class_short": CLASS_SHORT_NAMES[c_idx],
                        }
                    )

        if len(records) >= 50:
            print(
                f"[Dataset] Successfully discovered {len(records)} cervical cells across 7 classes in: {root}"
            )
            return pd.DataFrame(records)

    print(
        "[Dataset] No existing cervical cells found. Generating synthetic cytology cell fallback..."
    )
    fallback_dir = Path("./data/synthetic_herlev_cervical_cells")
    return generate_synthetic_cervical_cells(fallback_dir, samples_per_class=25)


def split_holdout_test_set(
    df: pd.DataFrame,
    test_split: float = 0.15,
    seed: int = 42,
    subsample: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits an unseen holdout test set (default 15%) from the training corpus."""
    if subsample and subsample < len(df):
        df, _ = train_test_split(
            df,
            train_size=subsample,
            stratify=df["label_idx"],
            random_state=seed,
        )

    train_df, test_df = train_test_split(
        df,
        test_size=test_split,
        stratify=df["label_idx"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def partition_iid(train_df: pd.DataFrame, num_clients: int, seed: int = 42) -> List[pd.DataFrame]:
    """Splits training data into uniform IID partitions across K clinic clients."""
    shuffled_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled_df)
    splits = []
    chunk_size = int(np.ceil(n / num_clients))
    for i in range(num_clients):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        if start < n:
            splits.append(shuffled_df.iloc[start:end].reset_index(drop=True))
        else:
            splits.append(pd.DataFrame(columns=train_df.columns))
    return splits


def partition_dirichlet_non_iid(
    train_df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """
    Partitions data using a Dirichlet distribution Dir(alpha) to simulate real-world
    heterogeneous cervical screening distributions across regional hospital clinics.
    """
    np.random.seed(seed)
    num_classes = len(CLASS_NAMES)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = train_df[train_df["label_idx"] == c].index.to_numpy()
        np.random.shuffle(idx_c)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        split_indices = np.split(idx_c, proportions)

        for client_id in range(num_clients):
            client_indices[client_id].extend(split_indices[client_id])

    client_dfs = []
    for cid in range(num_clients):
        c_df = (
            train_df.iloc[client_indices[cid]]
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        client_dfs.append(c_df)

    return client_dfs


def prepare_cervical_partitions(cfg: Dict) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Loads dataset, reserves central holdout test set, and partitions across Flower clinic clients."""
    data_path = cfg.get("data_path", "data")
    full_df = discover_cervical_dataset(data_path)

    test_split = float(cfg.get("test_split", 0.15))
    seed = int(cfg.get("seed", 42))
    subsample = cfg.get("subsample", None)
    if subsample is not None:
        subsample = int(subsample)

    train_df, test_df = split_holdout_test_set(
        full_df, test_split=test_split, seed=seed, subsample=subsample
    )

    num_clients = int(cfg.get("num_clients", 5))
    non_iid = bool(cfg.get("non_iid", False))
    alpha = float(cfg.get("dirichlet_alpha", 0.5))

    if non_iid:
        client_dfs = partition_dirichlet_non_iid(
            train_df, num_clients=num_clients, alpha=alpha, seed=seed
        )
    else:
        client_dfs = partition_iid(train_df, num_clients=num_clients, seed=seed)

    print(f"\n[Flower-Cervical] Total cells: {len(full_df)} | Holdout test set: {len(test_df)}")
    print(
        f"[Flower-Cervical] Partitioned across {num_clients} Cytology Clinic Clients (Non-IID: {non_iid})"
    )
    for cid, cdf in enumerate(client_dfs):
        counts = Counter(cdf["label_idx"])
        active_classes = len(counts)
        print(
            f"  • Clinic Client {cid}: {len(cdf):4d} cells, {active_classes}/7 classes represented"
        )

    return client_dfs, test_df

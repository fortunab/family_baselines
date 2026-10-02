"""
Herlev Cervical Cytology Dataset Loader & Non-IID Dirichlet Partitioner for Substra.
Handles 7-class Pap smear single-cell classification, ground-truth mask exclusion,
stratified 15% holdout test set generation, and simulated clinic partitions.
"""

import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
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
    "N.Sup",
    "N.Int",
    "N.Col",
    "Mild.Dys",
    "Mod.Dys",
    "Sev.Dys",
    "CIS",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

CANDIDATE_SEARCH_PATHS = [
    Path("data"),
    Path("../cervical_cells_federated_skorch/data"),
    Path("../cervical_cells_federated_flower/data"),
    Path("../herlev_cervical_baseline/data"),
    Path("../herlev_original_mde_baseline/data"),
    Path("C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/data"),
]


def match_class_name(folder_or_filename: str) -> str | None:
    """Matches directory or file name against 7 Herlev cervical classes."""
    name_clean = folder_or_filename.lower().replace("-", "_").replace(" ", "_")
    for cls in CLASS_NAMES:
        if cls.lower() in name_clean:
            return cls

    keyword_map = {
        "normal_superficiel": "01_normal_superficiel",
        "superficiel": "01_normal_superficiel",
        "normal_intermediate": "02_normal_intermediate",
        "intermediate": "02_normal_intermediate",
        "normal_columnar": "03_normal_columnar",
        "columnar": "03_normal_columnar",
        "light_dysplastic": "04_light_dysplastic",
        "mild_dysplastic": "04_light_dysplastic",
        "moderate_dysplastic": "05_moderate_dysplastic",
        "mod_dysplastic": "05_moderate_dysplastic",
        "severe_dysplastic": "06_severe_dysplastic",
        "sev_dysplastic": "06_severe_dysplastic",
        "carcinoma_in_situ": "07_carcinoma_in_situ",
        "carcinoma": "07_carcinoma_in_situ",
    }
    for kw, target_cls in keyword_map.items():
        if kw in name_clean:
            return target_cls
    return None


def generate_synthetic_cervical_dataset(base_dir: Path, samples_per_class: int = 35) -> Path:
    """Generates synthetic cervical cytology cellular patches for dry-run verification."""
    print(f"[Dataset] Generating synthetic cervical cytology patches in: {base_dir}...")
    base_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1337)  # Safe non-42 internal generator seed

    class_colors = [
        ((230, 210, 240), (140, 60, 160)),
        ((220, 200, 235), (120, 50, 140)),
        ((210, 190, 225), (100, 40, 120)),
        ((200, 180, 220), (90, 30, 110)),
        ((190, 170, 210), (80, 20, 100)),
        ((180, 160, 200), (70, 10, 90)),
        ((170, 150, 190), (60, 5, 80)),
    ]

    for c_idx, class_name in enumerate(CLASS_NAMES):
        cls_folder = base_dir / class_name
        cls_folder.mkdir(parents=True, exist_ok=True)
        cyto_color, nuc_color = class_colors[c_idx]

        for i in range(samples_per_class):
            img = Image.new("RGB", (224, 224), color=(245, 245, 250))
            draw = ImageDraw.Draw(img)

            # Draw cytoplasm ellipse
            cx, cy = 112 + rng.randint(-10, 10), 112 + rng.randint(-10, 10)
            rx, ry = rng.randint(65, 85), rng.randint(60, 80)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=cyto_color)

            # Draw nucleus
            nrx, nry = rng.randint(18 + c_idx * 3, 26 + c_idx * 4), rng.randint(18 + c_idx * 3, 26 + c_idx * 4)
            draw.ellipse([cx - nrx, cy - nry, cx + nrx, cy + nry], fill=nuc_color)

            img.save(cls_folder / f"synth_cell_{i:03d}.bmp")

    return base_dir


_DISCOVERED_DF_CACHE: pd.DataFrame | None = None


def discover_cervical_dataset(custom_path: str | None = None) -> pd.DataFrame:
    """Scans filesystem for Herlev Pap smear cells, strictly excluding mask files."""
    global _DISCOVERED_DF_CACHE
    if _DISCOVERED_DF_CACHE is not None and custom_path is None:
        return _DISCOVERED_DF_CACHE.copy()

    valid_paths: list[Path] = []
    if custom_path and str(custom_path).strip():
        valid_paths.append(Path(custom_path))
    valid_paths.extend([Path(p) for p in CANDIDATE_SEARCH_PATHS])

    data_dir: Path | None = None
    for p in valid_paths:
        if p.exists() and p.is_dir():
            data_dir = p
            break

    if data_dir is None:
        fallback_dir = Path("data/synthetic_cervical_herlev").resolve()
        data_dir = generate_synthetic_cervical_dataset(fallback_dir)

    rows = []
    for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for f in data_dir.rglob(ext):
            fname = f.name.lower()
            if fname.endswith(("-d.bmp", "-cyt.bmp")) or "_mask" in fname:
                continue

            cls = match_class_name(f.parent.name) or match_class_name(f.name)
            if cls is not None:
                rows.append({"filepath": str(f.resolve()), "label": cls})

    df = pd.DataFrame(rows)
    if df.empty or len(df["label"].unique()) < 2:
        print("[Dataset] Insufficient real cells found. Generating synthetic fallback dataset...")
        fallback_dir = Path("data/synthetic_cervical_herlev").resolve()
        generate_synthetic_cervical_dataset(fallback_dir)
        return discover_cervical_dataset(str(fallback_dir))

    print(f"[Dataset] Discovered {len(df)} cervical cells across {df['label'].nunique()} classes in: {data_dir}")
    _DISCOVERED_DF_CACHE = df.copy()
    return df


def partition_dirichlet_non_iid(
    train_df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[pd.DataFrame]:
    """Partitions dataset across hospital clients using Dirichlet non-IID label skew."""
    np.random.seed(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for class_name in CLASS_NAMES:
        class_idxs = train_df[train_df["label"] == class_name].index.tolist()
        if not class_idxs:
            continue
        np.random.shuffle(class_idxs)

        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        split_counts = (proportions * len(class_idxs)).astype(int)

        # Distribute remaining samples due to rounding
        diff = len(class_idxs) - split_counts.sum()
        for i in range(diff):
            split_counts[i % num_clients] += 1

        curr = 0
        for client_id, count in enumerate(split_counts):
            client_indices[client_id].extend(class_idxs[curr : curr + count])
            curr += count

    return [train_df.loc[indices].reset_index(drop=True) for indices in client_indices]


def partition_iid(
    train_df: pd.DataFrame,
    num_clients: int,
    seed: int,
) -> list[pd.DataFrame]:
    """Partitions dataset uniformly across clients (IID distribution)."""
    shuffled_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_indices = np.array_split(np.arange(len(shuffled_df)), num_clients)
    return [shuffled_df.iloc[idx].reset_index(drop=True) for idx in split_indices]


def extract_tensors_from_df(
    df: pd.DataFrame,
    img_size: int = 224,
    augment: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a DataFrame of image paths and labels into PyTorch tensors for skorch."""
    if len(df) == 0:
        return torch.empty((0, 3, img_size, img_size)), torch.empty((0,), dtype=torch.long)

    transform_list = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        transform_list = [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    transform = T.Compose(transform_list)

    images = []
    labels = []
    for _, row in df.iterrows():
        try:
            img = Image.open(row["filepath"]).convert("RGB")
            tensor = transform(img)
            images.append(tensor)
            labels.append(CLASS_TO_IDX[row["label"]])
        except Exception:
            continue

    if not images:
        return torch.empty((0, 3, img_size, img_size)), torch.empty((0,), dtype=torch.long)

    X = torch.stack(images).float()
    y = torch.tensor(labels, dtype=torch.long)
    return X, y


def prepare_cervical_partitions(
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Prepares stratified 15% holdout test set and client partitions using the given seed."""
    num_clients = int(cfg.get("num_clients", 5))
    is_non_iid = bool(cfg.get("non_iid", False))
    alpha = float(cfg.get("dirichlet_alpha", 0.5))
    test_split = float(cfg.get("test_split", 0.15))
    subsample = cfg.get("subsample", None)

    full_df = discover_cervical_dataset(cfg.get("data_path", None))

    if subsample and int(subsample) > 0:
        subsample_n = min(int(subsample), len(full_df))
        samples_per_cls = max(3, subsample_n // 7)
        dfs = []
        for _, g in full_df.groupby("label"):
            dfs.append(g.sample(min(len(g), samples_per_cls), random_state=seed))
        full_df = pd.concat(dfs, ignore_index=True)

    # Stratified split to ensure all classes in 15% holdout test set
    counts = Counter(full_df["label"])
    min_samples = min(counts.values()) if counts else 0
    test_n = max(7, int(len(full_df) * test_split))

    if min_samples >= 2 and test_n >= 7:
        train_df, test_df = train_test_split(
            full_df,
            test_size=test_split,
            stratify=full_df["label"],
            random_state=seed,
        )
    else:
        train_df, test_df = train_test_split(full_df, test_size=test_split, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if is_non_iid:
        client_dfs = partition_dirichlet_non_iid(train_df, num_clients, alpha=alpha, seed=seed)
    else:
        client_dfs = partition_iid(train_df, num_clients, seed=seed)

    return client_dfs, test_df

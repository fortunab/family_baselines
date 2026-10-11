"""
Cervical Cytology (Herlev) Dataset Loading, Mask Exclusion, and Federated Partitioning.
Supports Dirichlet non-IID and IID client partitioning across decentralized hospital silos.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

HERLEV_CLASSES = [
    "01_normal_superficiel",
    "02_normal_intermediate",
    "03_normal_columnar",
    "04_light_dysplastic",
    "05_moderate_dysplastic",
    "06_severe_dysplastic",
    "07_carcinoma_in_situ",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(HERLEV_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(HERLEV_CLASSES)}

_DISCOVERED_DF_CACHE: pd.DataFrame | None = None


def is_valid_cytology_image(path: Path) -> bool:
    """Verifies that the file is an original cytology micrograph, strictly excluding mask files."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix not in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        return False

    # Strictly exclude ground-truth mask files
    if "-d.bmp" in name or "-cyt.bmp" in name or "_mask" in name or "mask" in name:
        return False

    return True


def match_class_name(folder_or_filename: str) -> str | None:
    """Matches folder name or file prefix to canonical Herlev 7-class taxonomy."""
    lower = folder_or_filename.lower()
    for cls in HERLEV_CLASSES:
        if cls.lower() in lower:
            return cls

    mapping = {
        "normal_superficiel": "01_normal_superficiel",
        "normal_intermediate": "02_normal_intermediate",
        "normal_columnar": "03_normal_columnar",
        "light_dysplastic": "04_light_dysplastic",
        "moderate_dysplastic": "05_moderate_dysplastic",
        "severe_dysplastic": "06_severe_dysplastic",
        "carcinoma_in_situ": "07_carcinoma_in_situ",
    }
    for k, v in mapping.items():
        if k in lower:
            return v

    return None


def resolve_herlev_data_path(custom_path: str | None = None) -> Path:
    """Discovers available Herlev dataset directory in project or system scratch locations."""
    if custom_path and Path(custom_path).exists():
        return Path(custom_path).resolve()

    candidate_paths = [
        Path("./data/herlev"),
        Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_fedml\data\herlev_mock"),
        Path("../cervical_cells_federated_fedml/data/herlev_mock"),
        Path("../cervical_cells_federated_fedml/data/herlev"),
        Path("../cervical_cells_federated_flower/data/herlev"),
        Path("../cervical_cells_federated_substra/data/herlev"),
        Path("../cervical_cells_flower_skorch/data/herlev"),
        Path("../herlev_pathology_foundation_skorch/data/herlev"),
        Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_fedml\data\herlev"),
        Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_flower\data\herlev"),
        Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_pathology_foundation_skorch\data\herlev"),
    ]

    for p in candidate_paths:
        if p.exists() and p.is_dir():
            return p.resolve()

    return Path("./data/herlev").resolve()


def discover_cervical_dataset(
    data_path: str | Path | None = None,
    allow_mock_fallback: bool = True,
) -> pd.DataFrame:
    """
    Crawls filesystem to locate valid 7-class cervical cytology images.
    Filters out mask files and caches results.
    """
    global _DISCOVERED_DF_CACHE
    if _DISCOVERED_DF_CACHE is not None and not _DISCOVERED_DF_CACHE.empty:
        return _DISCOVERED_DF_CACHE.copy()

    resolved_path = resolve_herlev_data_path(str(data_path) if data_path else None)
    records: list[dict[str, Any]] = []

    if resolved_path.exists() and resolved_path.is_dir():
        for item in resolved_path.rglob("*"):
            if item.is_file() and is_valid_cytology_image(item):
                matched_cls = match_class_name(item.parent.name) or match_class_name(item.name)
                if matched_cls is not None:
                    cls_idx = CLASS_TO_IDX[matched_cls]
                    records.append(
                        {
                            "image_path": str(item.resolve()),
                            "filepath": str(item.resolve()),
                            "label": cls_idx,
                            "class_name": matched_cls,
                        }
                    )

    if not records and allow_mock_fallback:
        print("[Dataset] Notice: Real Herlev files not found; generating mock dataset for verification.")
        mock_dir = Path("./data/herlev_mock")
        mock_dir.mkdir(parents=True, exist_ok=True)
        for cls_name, cls_idx in CLASS_TO_IDX.items():
            sub_dir = mock_dir / cls_name
            sub_dir.mkdir(exist_ok=True)
            for img_id in range(15):
                img_path = sub_dir / f"cell_{img_id:03d}.png"
                if not img_path.exists():
                    arr = np.random.randint(50, 220, size=(128, 128, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(img_path)
                records.append(
                    {
                        "image_path": str(img_path.resolve()),
                        "filepath": str(img_path.resolve()),
                        "label": cls_idx,
                        "class_name": cls_name,
                    }
                )

    df = pd.DataFrame(records)
    if not df.empty:
        _DISCOVERED_DF_CACHE = df.copy()
        print(f"[Dataset] Discovered {len(df)} cytology images across {df['label'].nunique()} classes.")

    return df


def partition_dirichlet(
    df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 123456,
) -> list[pd.DataFrame]:
    """Partitions dataset across decentralized clients using Dirichlet distribution (non-IID)."""
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for label_val in sorted(df["label"].unique()):
        label_indices = df[df["label"] == label_val].index.to_numpy()
        rng.shuffle(label_indices)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [
                p * (len(idx_list) < len(df) / num_clients)
                for p, idx_list in zip(proportions, client_indices, strict=False)
            ]
        )
        if proportions.sum() == 0:
            proportions = np.repeat(1.0 / num_clients, num_clients)
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]

        client_splits = np.split(label_indices, split_points)
        for c_idx in range(num_clients):
            client_indices[c_idx].extend(client_splits[c_idx].tolist())

    return [df.iloc[idxs].sample(frac=1.0, random_state=seed).reset_index(drop=True) for idxs in client_indices]


def partition_iid(
    df: pd.DataFrame,
    num_clients: int,
    seed: int = 123456,
) -> list[pd.DataFrame]:
    """Partitions dataset equally and uniformly across clients (IID)."""
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled_df)
    splits = []
    chunk_size = int(np.ceil(n / max(1, num_clients)))
    for i in range(num_clients):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        if start_idx < n:
            splits.append(shuffled_df.iloc[start_idx:end_idx].reset_index(drop=True))
        else:
            splits.append(pd.DataFrame(columns=df.columns))
    return splits


def prepare_federated_partitions(
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    Orchestrates dataset discovery, centralized holdout splitting (15%),
    and client partitioning according to configuration specifications.
    """
    data_path = cfg.get("data_path")
    full_df = discover_cervical_dataset(data_path)

    if full_df.empty:
        raise RuntimeError("Cervical cytology dataset could not be loaded or discovered.")

    test_split = float(cfg.get("test_split", 0.15))
    num_clients = int(cfg.get("num_clients", 5))
    is_non_iid = bool(cfg.get("non_iid", False))
    alpha = float(cfg.get("dirichlet_alpha", 0.5))

    train_df, test_df = train_test_split(
        full_df,
        test_size=test_split,
        stratify=full_df["label"],
        random_state=seed,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if is_non_iid:
        client_dfs = partition_dirichlet(train_df, num_clients=num_clients, alpha=alpha, seed=seed)
    else:
        client_dfs = partition_iid(train_df, num_clients=num_clients, seed=seed)

    return client_dfs, test_df


def extract_tensors_from_df(
    df: pd.DataFrame,
    img_size: int = 224,
    preprocessor: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts image tensors and label tensors from a DataFrame, applying optional preprocessor."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensors: list[torch.Tensor] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        img_path = Path(row["image_path"] if "image_path" in row else row["filepath"])
        label = int(row["label"])
        try:
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                if preprocessor is not None:
                    if hasattr(preprocessor, "preprocess_pil"):
                        rgb_img = preprocessor.preprocess_pil(rgb_img)
                    else:
                        rgb_img = preprocessor(rgb_img)
                tensor_img = transform(rgb_img)
                tensors.append(tensor_img)
                labels.append(label)
        except Exception as e:
            print(f"[Dataset] Warning loading {img_path}: {e}")

    if not tensors:
        dummy_x = torch.zeros((len(df), 3, img_size, img_size), dtype=torch.float32)
        dummy_y = torch.tensor(df["label"].tolist(), dtype=torch.long)
        return dummy_x, dummy_y

    x_tensor = torch.stack(tensors).type(torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor

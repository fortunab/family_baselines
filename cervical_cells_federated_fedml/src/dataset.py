"""
Herlev 7-Class Cervical Cytology Dataset Loader & FedML Partitioning Pipeline.
Handles mask exclusion, stratified holdout splitting, Dirichlet/IID partitioning,
and fast tensor extraction for skorch and fastai.
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

CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(HERLEV_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(HERLEV_CLASSES)}

DEFAULT_HERLEV_CANDIDATE_PATHS = [
    Path(
        r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_substra\data\synthetic_cervical_herlev"
    ),
    Path(r"..\cervical_cells_federated_substra\data\synthetic_cervical_herlev"),
    Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_flower\data\herlev"),
    Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_substra\data\herlev"),
    Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_fastai_skorch_env_benchmark\data\herlev"),
    Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_cervical_baseline\data"),
    Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\cervical_cells_federated_fedml\data\herlev"),
    Path("./data/herlev"),
]

# In-memory discovery cache to accelerate K-candidate seed search
_DISCOVERED_DF_CACHE: pd.DataFrame | None = None


def match_class_name(name: str) -> str | None:
    """Matches directory or file name against 7 Herlev cervical classes."""
    n = name.lower().replace("-", "_").replace(" ", "_")
    if "superficiel" in n:
        return "01_normal_superficiel"
    if "intermediate" in n:
        return "02_normal_intermediate"
    if "columnar" in n:
        return "03_normal_columnar"
    if "light_dysplastic" in n or "mild_dysplasia" in n or "mild_dysplastic" in n or "light" in n:
        return "04_light_dysplastic"
    if "moderate" in n or "mod_dysplastic" in n:
        return "05_moderate_dysplastic"
    if "severe" in n or "sev_dysplastic" in n:
        return "06_severe_dysplastic"
    if "carcinoma" in n or "cis" in n:
        return "07_carcinoma_in_situ"
    return None


def resolve_herlev_data_path(custom_path: str | None = None) -> Path:
    """Finds the first existing directory containing Herlev dataset images."""
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_dir():
            return p

    for cand in DEFAULT_HERLEV_CANDIDATE_PATHS:
        if cand.exists() and cand.is_dir():
            subdirs = [d.name for d in cand.iterdir() if d.is_dir()]
            matches = [match_class_name(s) for s in subdirs if match_class_name(s) is not None]
            if len(set(matches)) >= 2:
                return cand

    return DEFAULT_HERLEV_CANDIDATE_PATHS[0]


def is_valid_cytology_image(file_path: Path) -> bool:
    """
    Validates whether an image file is a true cytology micrograph or a mask.
    Excludes ground-truth segmentation masks (-d.bmp, -cyt.bmp, *_mask*).
    """
    lower_name = file_path.name.lower()
    if lower_name.endswith(("-d.bmp", "-cyt.bmp")):
        return False
    if "_mask" in lower_name or "-mask" in lower_name:
        return False
    return file_path.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def discover_cervical_dataset(
    data_path: Path | str | None = None,
    allow_mock_fallback: bool = True,
) -> pd.DataFrame:
    """
    Scans the directory structure for 7 Herlev classes and builds a catalog DataFrame.
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
    """
    Partitions the dataset across decentralized clients using a Dirichlet distribution (non-IID).
    """
    rng = np.random.default_rng(seed)
    num_classes = len(HERLEV_CLASSES)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for cls_idx in range(num_classes):
        cls_rows = df[df["label"] == cls_idx].index.to_numpy()
        if len(cls_rows) == 0:
            continue
        rng.shuffle(cls_rows)

        # Sample Dirichlet proportions
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()

        # Split indices
        split_points = (np.cumsum(proportions) * len(cls_rows)).astype(int)[:-1]
        splits = np.split(cls_rows, split_points)

        for c_idx in range(num_clients):
            client_indices[c_idx].extend(splits[c_idx].tolist())

    client_dfs = []
    for c_idx in range(num_clients):
        idxs = client_indices[c_idx]
        if len(idxs) == 0:
            # Fallback to random sample if Dirichlet left client empty
            idxs = rng.choice(df.index.to_numpy(), size=max(2, len(df) // (num_clients * 2)), replace=False).tolist()
        client_dfs.append(df.loc[idxs].sample(frac=1.0, random_state=seed).reset_index(drop=True))

    return client_dfs


def partition_iid(
    df: pd.DataFrame,
    num_clients: int,
    seed: int = 123456,
) -> list[pd.DataFrame]:
    """Partitions the dataset uniformly across decentralized clients (IID)."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    splits = np.array_split(shuffled.index.to_numpy(), num_clients)
    return [shuffled.loc[split].reset_index(drop=True) for split in splits]


def prepare_fedml_partitions(
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

    # Stratified centralized test holdout
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts image tensors and label tensors from a DataFrame.
    Used for skorch neural network training and evaluation.
    """
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
                tensor_img = transform(rgb_img)
                tensors.append(tensor_img)
                labels.append(label)
        except Exception as e:
            print(f"[Dataset] Warning: Failed to load {img_path}: {e}")

    if not tensors:
        dummy_x = torch.zeros((len(df), 3, img_size, img_size), dtype=torch.float32)
        dummy_y = torch.tensor(df["label"].tolist(), dtype=torch.long)
        return dummy_x, dummy_y

    x_tensor = torch.stack(tensors).type(torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor

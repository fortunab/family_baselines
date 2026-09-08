"""
Colorectal Histology 8-Class Dataset Loader & Pipeline.
Supports:
- 8 Tissue Classes: TUMOR, STROMA, COMPLEX, LYMPHO, DEBRIS, MUCOSA, ADIPOSE, EMPTY
- Protocol: 70% Train / 15% Val / 15% Test with dynamic randomized seeds.
- Dual Outputs: fastai DataLoaders & skorch / NumPy Arrays.
"""

import os
import random
import secrets
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

CLASS_NAMES = [
    "01_TUMOR", "02_STROMA", "03_COMPLEX", "04_LYMPHO",
    "05_DEBRIS", "06_MUCOSA", "07_ADIPOSE", "08_EMPTY"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASS_NAMES)}


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None or seed == 0:
        seed = secrets.randbelow(900000) + 100000

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Random-Seed] Active experiment seed: {seed}")
    return seed


def generate_synthetic_histology_tiles(output_dir: Path, samples_per_class: int = 15) -> List[Tuple[str, str]]:
    print(f"[Dataset] Generating {samples_per_class * len(CLASS_NAMES)} synthetic histology tiles in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for cls_name in CLASS_NAMES:
        cls_dir = output_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = Image.new("RGB", (150, 150), color=(
                random.randint(190, 240),
                random.randint(140, 190),
                random.randint(180, 230)
            ))
            draw = ImageDraw.Draw(img)
            # Add cellular pattern dots
            for _ in range(40):
                x = random.randint(5, 140)
                y = random.randint(5, 140)
                r = random.randint(2, 6)
                draw.ellipse([(x, y), (x+r, y+r)], fill=(random.randint(50, 110), random.randint(20, 60), random.randint(90, 140)))

            file_path = cls_dir / f"tile_{i:04d}.tif"
            img.save(file_path)
            samples.append((str(file_path), cls_name))

    return samples


def find_or_prepare_dataset(data_dir: Path) -> List[Tuple[str, str]]:
    data_dir = Path(data_dir)
    
    # 1. Check direct data_dir
    existing_samples = []
    for cls_name in CLASS_NAMES:
        cls_folder = data_dir / cls_name
        if cls_folder.exists() and cls_folder.is_dir():
            for f in cls_folder.glob("*.*"):
                if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                    existing_samples.append((str(f), cls_name))

    if len(existing_samples) >= 80:
        print(f"[Dataset] Found {len(existing_samples)} valid images across 8 classes in: {data_dir}")
        return existing_samples

    # 2. Check sibling directories in workspace
    sibling_candidates = [
        data_dir.parent.parent / "colorectal_histology_svm_baseline" / "data" / "Kather_texture_2016_image_tiles_5000",
        data_dir.parent.parent / "colorectal_histology_tf_foundation" / "data" / "Kather_texture_2016_image_tiles_5000",
        data_dir.parent / "colorectal_histology_svm_baseline" / "data" / "Kather_texture_2016_image_tiles_5000"
    ]
    for cand in sibling_candidates:
        if cand.exists() and cand.is_dir():
            cand_samples = []
            for cls_name in CLASS_NAMES:
                c_dir = cand / cls_name
                if c_dir.exists():
                    for f in c_dir.glob("*.*"):
                        if f.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                            cand_samples.append((str(f), cls_name))
            if len(cand_samples) >= 80:
                print(f"[Dataset] Reusing {len(cand_samples)} images from existing dataset at: {cand}")
                return cand_samples

    # 3. Fallback: generate local demo tiles
    demo_dir = data_dir / "histology_tiles_demo"
    return generate_synthetic_histology_tiles(demo_dir, samples_per_class=20)


def create_split_dataframes(
    data_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None,
    subsample: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    active_seed = setup_random_seed(seed)
    samples = find_or_prepare_dataset(data_dir)

    df = pd.DataFrame(samples, columns=["filepath", "label"])
    df["label_idx"] = df["label"].map(CLASS_TO_IDX)

    if subsample is not None and subsample > 0 and subsample < len(df):
        df = df.sample(n=subsample, random_state=active_seed).reset_index(drop=True)
        print(f"[Dataset] Subsampled to {len(df)} total items for rapid execution.")

    holdout_ratio = val_split + test_split
    
    # Check if stratify is valid (at least 2 samples per class in the current dataframe)
    label_counts = df["label_idx"].value_counts()
    can_stratify_train = (label_counts.min() >= 2) and (len(df) >= 32)
    
    train_df, holdout_df = train_test_split(
        df, test_size=holdout_ratio, random_state=active_seed, stratify=df["label_idx"] if can_stratify_train else None
    )

    test_rel_ratio = test_split / holdout_ratio
    holdout_counts = holdout_df["label_idx"].value_counts()
    can_stratify_val = (holdout_counts.min() >= 2) and (len(holdout_df) >= 16)
    
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_rel_ratio, random_state=active_seed, stratify=holdout_df["label_idx"] if can_stratify_val else None
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    print(f"[Dataset] Partitioned: Train={len(train_df)} (70%), Val={len(val_df)} (15%), Test={len(test_df)} (15%)")
    return train_df, val_df, test_df, active_seed


def get_numpy_tensors(
    df: pd.DataFrame,
    image_size: int = 224
) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for _, row in df.iterrows():
        try:
            im = Image.open(row["filepath"]).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.float32) / 255.0
            # Transpose to [C, H, W] for PyTorch / skorch
            arr = np.transpose(arr, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            arr = (arr - mean) / std
            images.append(arr)
            labels.append(row["label_idx"])
        except Exception:
            continue

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

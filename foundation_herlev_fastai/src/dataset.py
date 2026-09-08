"""
Herlev Pap Smear Cervical Cytology Dataset Loader & Dataframe Engine for fastai.
Strictly excludes ground-truth segmentation masks (-d.bmp, -cyt.bmp, *_mask.*).
Supports 7-Class Dysplasia & Carcinoma Grading:
1. 01_normal_superficiel
2. 02_normal_intermediate
3. 03_normal_columnar
4. 04_light_dysplastic (mild dysplasia / CIN 1 / LSIL)
5. 05_moderate_dysplastic (moderate dysplasia / CIN 2 / HSIL)
6. 06_severe_dysplastic (severe dysplasia / CIN 3 / HSIL)
7. 07_carcinoma_in_situ (carcinoma in situ / invasive carcinoma)
"""

import random
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
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

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

CLASS_DESCRIPTIONS = {
    "01_normal_superficiel": "Normal Superficial Squamous (N.Sup)",
    "02_normal_intermediate": "Normal Intermediate Squamous (N.Int)",
    "03_normal_columnar": "Normal Columnar Endocervical (N.Col)",
    "04_light_dysplastic": "Light / Mild Dysplasia / CIN 1 / LSIL (Mild)",
    "05_moderate_dysplastic": "Moderate Dysplasia / CIN 2 / HSIL (Mod)",
    "06_severe_dysplastic": "Severe Dysplasia / CIN 3 / HSIL (Sev)",
    "07_carcinoma_in_situ": "Carcinoma in Situ / Malignant (CIS)",
}

CLASS_ALIASES = {
    0: [
        "01_normal_superficiel",
        "01_normal_superficial",
        "normal_superficiel",
        "normal_superficial",
        "superficiel",
        "superficial",
    ],
    1: ["02_normal_intermediate", "normal_intermediate", "intermediate"],
    2: ["03_normal_columnar", "normal_columnar", "columnar"],
    3: [
        "04_light_dysplastic",
        "04_mild_dysplastic",
        "light_dysplastic",
        "mild_dysplastic",
        "light",
        "mild",
    ],
    4: ["05_moderate_dysplastic", "moderate_dysplastic", "moderate", "mod_dysplastic"],
    5: ["06_severe_dysplastic", "severe_dysplastic", "severe", "sev_dysplastic"],
    6: ["07_carcinoma_in_situ", "carcinoma_in_situ", "carcinoma", "cis"],
}


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = secrets.randbelow(900000) + 100000
    random.seed(seed)
    np.random.seed(seed)
    print(f"[Dataset] Active random seed: {seed}")
    return seed


def is_valid_herlev_image(filename: str) -> bool:
    """Excludes ground truth segmentation masks (-d.bmp, -cyt.bmp, *_mask.*)."""
    name_lower = filename.lower()
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not name_lower.endswith(valid_exts):
        return False
    mask_markers = ["-d.bmp", "-cyt.bmp", "-d.png", "-cyt.png", "_mask", "-mask", "_label"]
    for marker in mask_markers:
        if marker in name_lower:
            return False
    return True


def create_synthetic_herlev_demo(data_dir: Path, num_samples_per_class: int = 35):
    print(
        f"[Dataset] Generating synthetic Herlev demo dataset in '{data_dir}' ({num_samples_per_class * 7} cells)..."
    )
    for class_idx, cname in enumerate(CLASS_NAMES):
        folder = data_dir / cname
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples_per_class):
            img_path = folder / f"cell_{i:04d}.bmp"
            if img_path.exists():
                continue

            base = Image.new(
                "RGB",
                (224, 224),
                color=(
                    random.randint(235, 250),
                    random.randint(235, 250),
                    random.randint(240, 255),
                ),
            )
            draw = ImageDraw.Draw(base)

            cx, cy = 112 + random.randint(-8, 8), 112 + random.randint(-8, 8)
            cyt_radius = max(40, 85 - class_idx * 6)
            cyt_color = (
                random.randint(180, 210) if class_idx > 3 else random.randint(160, 190),
                random.randint(200, 230) if class_idx < 4 else random.randint(180, 200),
                random.randint(210, 240),
            )
            draw.ellipse(
                [(cx - cyt_radius, cy - cyt_radius), (cx + cyt_radius, cy + cyt_radius)],
                fill=cyt_color,
                outline=(160, 180, 200),
            )

            nuc_radius = 14 + class_idx * 5 + random.randint(-2, 3)
            nuc_darkness = max(20, 90 - class_idx * 10)
            nuc_color = (
                nuc_darkness + random.randint(10, 30),
                nuc_darkness,
                nuc_darkness + random.randint(30, 60),
            )
            draw.ellipse(
                [(cx - nuc_radius, cy - nuc_radius), (cx + nuc_radius, cy + nuc_radius)],
                fill=nuc_color,
                outline=(30, 20, 50),
            )

            for _ in range(5 + class_idx * 4):
                px = cx + random.randint(-nuc_radius + 4, nuc_radius - 4)
                py = cy + random.randint(-nuc_radius + 4, nuc_radius - 4)
                draw.point((px, py), fill=(20, 10, 40))

            base = base.filter(ImageFilter.GaussianBlur(radius=0.7))
            base.save(img_path)


def index_directory_for_herlev(search_dir: Path) -> Tuple[List[str], List[str], List[int]]:
    search_dir = Path(search_dir).resolve()
    all_subdirs = [d for d in search_dir.rglob("*") if d.is_dir()]
    if not all_subdirs and search_dir.is_dir():
        all_subdirs = [d for d in search_dir.iterdir() if d.is_dir()]

    filepaths: List[str] = []
    labels: List[str] = []
    indices: List[int] = []
    class_counts = {idx: 0 for idx in range(7)}

    for class_idx in range(7):
        canonical_name = CLASS_NAMES[class_idx]
        aliases = CLASS_ALIASES[class_idx]
        matched_folders = []

        for folder in all_subdirs:
            fname = folder.name.lower().replace("-", "_").replace(" ", "_")
            if any(alias in fname for alias in aliases):
                matched_folders.append(folder)

        for folder in matched_folders:
            for f in folder.iterdir():
                if f.is_file() and is_valid_herlev_image(f.name):
                    filepaths.append(str(f.resolve()))
                    labels.append(canonical_name)
                    indices.append(class_idx)
                    class_counts[class_idx] += 1

    return filepaths, labels, indices


def load_herlev_dataframe(
    data_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None,
    subsample: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], int]:
    active_seed = setup_random_seed(seed)
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    filepaths, labels, indices = index_directory_for_herlev(data_dir)

    # If no data found in data_dir, search candidate sibling directories
    if len(filepaths) == 0:
        sibling_candidates = [
            data_dir.parent.parent / "herlev_cervical_baseline" / "data" / "Herlev Dataset",
            data_dir.parent.parent / "herlev_cervical_baseline" / "data",
            data_dir.parent.parent / "herlev_original_mde_baseline" / "data",
        ]
        for candidate in sibling_candidates:
            if candidate.exists():
                print(f"[Dataset] Scanning sibling data directory: {candidate}")
                cand_paths, cand_labels, cand_indices = index_directory_for_herlev(candidate)
                if len(cand_paths) > 0:
                    filepaths, labels, indices = cand_paths, cand_labels, cand_indices
                    print(f"[Dataset] Found {len(filepaths)} valid Herlev cells in sibling path.")
                    break

    # If still empty or classes missing, generate synthetic demo set
    class_counts = {idx: 0 for idx in range(7)}
    for idx in indices:
        class_counts[idx] += 1

    if len(filepaths) == 0 or any(count == 0 for count in class_counts.values()):
        print(f"[Dataset] No full Herlev dataset found. Creating demo in {data_dir}...")
        create_synthetic_herlev_demo(data_dir, num_samples_per_class=35)
        filepaths, labels, indices = index_directory_for_herlev(data_dir)

    full_df = pd.DataFrame(
        {
            "filepath": filepaths,
            "label": labels,
            "class_idx": indices,
        }
    )

    if subsample is not None and 0 < subsample < len(full_df):
        print(f"[Dataset] Subsampling dataset to {subsample} samples (seed={active_seed})...")
        samples_per_class = max(2, int(np.ceil(subsample / 7)))
        sub_dfs = []
        for c_idx in range(7):
            c_df = full_df[full_df["class_idx"] == c_idx]
            if len(c_df) > 0:
                n_draw = min(len(c_df), samples_per_class)
                sampled = c_df.sample(
                    n_draw,
                    random_state=active_seed,
                    replace=False,
                )
                sub_dfs.append(sampled)
        if sub_dfs:
            full_df = pd.concat(sub_dfs, ignore_index=True)
            full_df = full_df.sample(frac=1.0, random_state=active_seed).reset_index(drop=True)

    print(
        f"[Dataset] Indexed {len(full_df)} Herlev cytology images across 7 classes (Seed={active_seed}):"
    )
    for cname in CLASS_NAMES:
        count = len(full_df[full_df["label"] == cname])
        print(f"  • {cname:<26}: {count} cells")

    holdout_ratio = val_split + test_split
    min_count_full = full_df["class_idx"].value_counts().min()
    stratify_full = full_df["class_idx"] if min_count_full >= 2 else None

    train_df, holdout_df = train_test_split(
        full_df,
        test_size=holdout_ratio,
        stratify=stratify_full,
        random_state=active_seed,
    )

    min_count_holdout = holdout_df["class_idx"].value_counts().min()
    stratify_holdout = holdout_df["class_idx"] if min_count_holdout >= 2 else None

    test_rel_ratio = test_split / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_rel_ratio,
        stratify=stratify_holdout,
        random_state=active_seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(
        f"[Dataset] Splits: Train={len(train_df)} (70%), Val={len(val_df)} (15%), Test={len(test_df)} (15%)"
    )

    return train_df, val_df, test_df, CLASS_NAMES, active_seed

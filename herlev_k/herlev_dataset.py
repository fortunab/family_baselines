"""
Herlev Cervical Cytology Dataset Loader & Augmentation Engine.
Natively compatible with the official Kaggle dataset:
https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset

Supports 7-class Pap smear single-cell dysplasia & carcinoma grading:
1. normal_superficial
2. normal_intermediate
3. normal_columnar
4. mild_dysplastic (light_dysplastic)
5. moderate_dysplastic
6. severe_dysplastic
7. carcinoma_in_situ

Features:
- Filters out mask files (-d.bmp, -cyt.bmp, *_mask.*)
- Fuzzy folder name matching for all Kaggle variants
- 70/15/15 train/val/test splits with dynamic random seeds
"""

import os
import sys
import random
import secrets
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "01_normal_superficial",
    "02_normal_intermediate",
    "03_normal_columnar",
    "04_mild_dysplastic",
    "05_moderate_dysplastic",
    "06_severe_dysplastic",
    "07_carcinoma_in_situ"
]

CLASS_DESCRIPTIONS = {
    "01_normal_superficial": "Normal Superficial Squamous (N.Sup)",
    "02_normal_intermediate": "Normal Intermediate Squamous (N.Int)",
    "03_normal_columnar": "Normal Columnar Endocervical (N.Col)",
    "04_mild_dysplastic": "Mild Dysplasia / CIN 1 / LSIL (Mild)",
    "05_moderate_dysplastic": "Moderate Dysplasia / CIN 2 / HSIL (Mod)",
    "06_severe_dysplastic": "Severe Dysplasia / CIN 3 / HSIL (Sev)",
    "07_carcinoma_in_situ": "Carcinoma in Situ / Malignant (CIS)"
}

# Fuzzy folder aliases matching various Kaggle extraction layouts
CLASS_ALIASES = {
    0: ["01_normal_superficial", "normal_superficial", "normal_superficiel", "superficial", "superficiel"],
    1: ["02_normal_intermediate", "normal_intermediate", "intermediate"],
    2: ["03_normal_columnar", "normal_columnar", "columnar"],
    3: ["04_mild_dysplastic", "mild_dysplastic", "light_dysplastic", "mild", "light"],
    4: ["05_moderate_dysplastic", "moderate_dysplastic", "moderate", "mod_dysplastic"],
    5: ["06_severe_dysplastic", "severe_dysplastic", "severe", "sev_dysplastic"],
    6: ["07_carcinoma_in_situ", "carcinoma_in_situ", "carcinoma", "cis"]
}


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = secrets.randbelow(900000) + 100000
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[Random-Seed] Active experiment seed: {seed}")
    return seed


class HerlevCytologyDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        label = int(self.labels[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_cytology_transforms(
    img_size: int = 224,
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def is_valid_cytology_image(filename: str) -> bool:
    """
    Excludes ground truth segmentation masks (-d.bmp, -cyt.bmp, *_mask.*) from Kaggle dataset.
    """
    name_lower = filename.lower()
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    if not name_lower.endswith(valid_exts):
        return False
    
    # Herlev segmentation mask markers
    mask_markers = ["-d.bmp", "-cyt.bmp", "-d.png", "-cyt.png", "_mask", "-mask", "_label"]
    for marker in mask_markers:
        if marker in name_lower:
            return False
    return True


def create_synthetic_demo_cytology(data_dir: Path, num_samples_per_class: int = 35):
    print(f"[Dataset] Creating Herlev demo dataset in {data_dir} ({num_samples_per_class*7} cells)...")
    for class_idx, cname in enumerate(CLASS_NAMES):
        folder = data_dir / cname
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples_per_class):
            img_path = folder / f"cell_{i:04d}.bmp"
            if img_path.exists():
                continue

            base = Image.new("RGB", (200, 200), color=(
                random.randint(235, 250),
                random.randint(235, 250),
                random.randint(240, 255)
            ))
            draw = ImageDraw.Draw(base)

            cx, cy = 100 + random.randint(-8, 8), 100 + random.randint(-8, 8)
            cyt_radius = max(35, 75 - class_idx * 5)
            cyt_color = (
                random.randint(180, 210) if class_idx > 3 else random.randint(160, 190),
                random.randint(200, 230) if class_idx < 4 else random.randint(180, 200),
                random.randint(210, 240)
            )
            draw.ellipse(
                [(cx - cyt_radius, cy - cyt_radius), (cx + cyt_radius, cy + cyt_radius)],
                fill=cyt_color,
                outline=(160, 180, 200)
            )

            nuc_radius = 12 + class_idx * 5 + random.randint(-2, 3)
            nuc_darkness = max(20, 90 - class_idx * 10)
            nuc_color = (
                nuc_darkness + random.randint(10, 30),
                nuc_darkness,
                nuc_darkness + random.randint(30, 60)
            )
            draw.ellipse(
                [(cx - nuc_radius, cy - nuc_radius), (cx + nuc_radius, cy + nuc_radius)],
                fill=nuc_color,
                outline=(30, 20, 50)
            )

            for _ in range(5 + class_idx * 3):
                px = cx + random.randint(-nuc_radius + 4, nuc_radius - 4)
                py = cy + random.randint(-nuc_radius + 4, nuc_radius - 4)
                draw.point((px, py), fill=(20, 10, 40))

            base = base.filter(ImageFilter.GaussianBlur(radius=0.7))
            base.save(img_path)


def load_herlev_dataset_paths_and_labels(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find all subdirectories recursively
    all_subdirs = [d for d in data_dir.rglob("*") if d.is_dir()]
    if not all_subdirs:
        all_subdirs = [d for d in data_dir.iterdir() if d.is_dir()]

    image_paths, labels = [], []
    class_counts = {idx: 0 for idx in range(7)}

    for class_idx in range(7):
        aliases = CLASS_ALIASES[class_idx]
        matched_folders = []

        for folder in all_subdirs:
            fname = folder.name.lower().replace("-", "_").replace(" ", "_")
            if any(alias in fname for alias in aliases):
                matched_folders.append(folder)

        for folder in matched_folders:
            for f in folder.iterdir():
                if f.is_file() and is_valid_cytology_image(f.name):
                    image_paths.append(str(f.resolve()))
                    labels.append(class_idx)
                    class_counts[class_idx] += 1

    # If no real data found or incomplete classes, create synthetic demo
    if len(image_paths) == 0 or any(count == 0 for count in class_counts.values()):
        print(f"[Dataset] No complete Kaggle folders found in '{data_dir}'. Generating demo dataset...")
        create_synthetic_demo_cytology(data_dir, num_samples_per_class=35)
        return load_herlev_dataset_paths_and_labels(data_dir)

    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[Dataset] Successfully indexed {len(image_paths)} Herlev cytology images across 7 classes:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  • {name:<26}: {class_counts[idx]} cells")

    return image_paths, labels_arr, CLASS_NAMES


def create_herlev_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: Optional[int] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    subsample: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    active_seed = setup_random_seed(seed)
    image_paths, labels, class_names = load_herlev_dataset_paths_and_labels(data_dir)

    if subsample is not None and subsample < len(image_paths):
        np.random.seed(active_seed)
        sub_idx = np.random.choice(len(image_paths), size=subsample, replace=False)
        image_paths = [image_paths[i] for i in sub_idx]
        labels = labels[sub_idx]

    holdout_ratio = val_split + test_split
    train_paths, holdout_paths, train_lbls, holdout_lbls = train_test_split(
        image_paths, labels, test_size=holdout_ratio, stratify=labels, random_state=active_seed
    )

    test_rel_ratio = test_split / holdout_ratio
    val_paths, test_paths, val_lbls, test_lbls = train_test_split(
        holdout_paths, holdout_lbls, test_size=test_rel_ratio, stratify=holdout_lbls, random_state=active_seed
    )

    print(f"[Dataset] Split sizes: Train={len(train_paths)} (70%), Val={len(val_paths)} (15%), Test={len(test_paths)} (15%)")

    train_tf = get_cytology_transforms(img_size=img_size, is_training=True, mean=mean, std=std)
    eval_tf = get_cytology_transforms(img_size=img_size, is_training=False, mean=mean, std=std)

    train_dataset = HerlevCytologyDataset(train_paths, train_lbls, transform=train_tf)
    val_dataset = HerlevCytologyDataset(val_paths, val_lbls, transform=eval_tf)
    test_dataset = HerlevCytologyDataset(test_paths, test_lbls, transform=eval_tf)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    return train_loader, val_loader, test_loader, class_names, active_seed


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_path = current_dir / "data"
    tr_loader, vl_loader, ts_loader, names, s = create_herlev_dataloaders(data_path, batch_size=16)
    for imgs, lbls in tr_loader:
        print(f"[Sanity Test] Batch tensor shape: {imgs.shape}, Labels: {lbls.shape}, Seed: {s}")
        break

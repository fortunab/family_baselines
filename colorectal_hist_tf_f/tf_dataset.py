"""
TensorFlow tf.data.Dataset Pipeline & Augmentation Engine for Colorectal Histology.
Supports 8 tissue classes from Kather et al. (5,000 H&E tiles):
1. 01_TUMOR
2. 02_STROMA
3. 03_COMPLEX
4. 04_LYMPHO
5. 05_DEBRIS
6. 06_MUCOSA
7. 07_ADIPOSE
8. 08_EMPTY

Universal loading supporting .tif, .tiff, .png, .jpg, .bmp.
Protocol: 70% Train / 15% Val / 15% Test stratified splits with dynamic random seeds.
"""

import os
import sys
import random
import secrets
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

CLASS_NAMES = [
    "01_TUMOR",
    "02_STROMA",
    "03_COMPLEX",
    "04_LYMPHO",
    "05_DEBRIS",
    "06_MUCOSA",
    "07_ADIPOSE",
    "08_EMPTY"
]

CLASS_DESCRIPTIONS = {
    "01_TUMOR": "Colorectal Adenocarcinoma Epithelium (TUM)",
    "02_STROMA": "Cancer-Associated Stroma (STR)",
    "03_COMPLEX": "Complex Stroma / Mixed Glands (COMP)",
    "04_LYMPHO": "Immune Cells / Lymphocytes (LYM)",
    "05_DEBRIS": "Necrotic Debris & Mucus (DEB)",
    "06_MUCOSA": "Normal Colon Mucosa (NORM)",
    "07_ADIPOSE": "Adipose Fat Tissue (ADI)",
    "08_EMPTY": "Background / Glass Slide (BACK)"
}


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = secrets.randbelow(900000) + 100000

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Random-Seed] Active experiment seed: {seed}")
    return seed


def create_synthetic_demo_histology(data_dir: Path, num_samples_per_class: int = 35):
    print(f"[Dataset] Creating Colorectal Histology demo dataset in {data_dir} ({num_samples_per_class*8} tiles)...")
    colors = {
        0: (180, 50, 160),   # Tumor (dark purple)
        1: (220, 150, 180),  # Stroma (pinkish fibrous)
        2: (200, 100, 170),  # Complex
        3: (100, 30, 130),   # Lympho (dense blue/violet dots)
        4: (160, 140, 150),  # Debris
        5: (230, 180, 200),  # Normal Mucosa
        6: (245, 245, 230),  # Adipose (white rings)
        7: (250, 250, 250)   # Empty (clean background)
    }

    for idx, cname in enumerate(CLASS_NAMES):
        folder = data_dir / cname
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples_per_class):
            img_path = folder / f"tile_{i:04d}.png"
            if img_path.exists():
                continue

            base_color = colors[idx]
            base = Image.new("RGB", (150, 150), color=(
                max(0, min(255, base_color[0] + random.randint(-15, 15))),
                max(0, min(255, base_color[1] + random.randint(-15, 15))),
                max(0, min(255, base_color[2] + random.randint(-15, 15)))
            ))
            draw = ImageDraw.Draw(base)

            if idx == 0:  # Tumor glands
                for _ in range(8):
                    gx, gy = random.randint(20, 130), random.randint(20, 130)
                    draw.ellipse([(gx-15, gy-15), (gx+15, gy+15)], fill=(120, 20, 100), outline=(80, 10, 70))
            elif idx == 3:  # Lymphocytes
                for _ in range(60):
                    lx, ly = random.randint(10, 140), random.randint(10, 140)
                    draw.ellipse([(lx-2, ly-2), (lx+2, ly+2)], fill=(40, 10, 80))
            elif idx == 6:  # Adipose mesh
                for _ in range(12):
                    ax, ay = random.randint(20, 130), random.randint(20, 130)
                    draw.ellipse([(ax-12, ay-12), (ax+12, ay+12)], fill=(255, 255, 250), outline=(210, 180, 190))

            base = base.filter(ImageFilter.GaussianBlur(radius=0.5))
            base.save(img_path)


def load_histology_paths_and_labels(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = [], []
    class_counts = {idx: 0 for idx in range(8)}

    for idx, cname in enumerate(CLASS_NAMES):
        folder = data_dir / cname
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(str(f.resolve()))
                    labels.append(idx)
                    class_counts[idx] += 1

    # Fallback to demo if empty
    if len(image_paths) == 0 or any(count == 0 for count in class_counts.values()):
        print(f"[Dataset] Creating demo dataset in '{data_dir}'...")
        create_synthetic_demo_histology(data_dir, num_samples_per_class=35)
        image_paths, labels = [], []
        class_counts = {idx: 0 for idx in range(8)}
        for idx, cname in enumerate(CLASS_NAMES):
            folder = data_dir / cname
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]:
                    image_paths.append(str(f.resolve()))
                    labels.append(idx)
                    class_counts[idx] += 1

    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[Dataset] Successfully indexed {len(image_paths)} histology tiles across 8 classes:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  • {name:<14}: {class_counts[idx]} tiles")

    return image_paths, labels_arr, CLASS_NAMES


def read_image_from_path(path_bytes: bytes) -> np.ndarray:
    path_str = path_bytes.decode('utf-8')
    img = Image.open(path_str).convert('RGB')
    return np.array(img, dtype=np.float32)


def build_tf_dataset(
    image_paths: List[str],
    labels: np.ndarray,
    img_size: int = 224,
    batch_size: int = 32,
    is_training: bool = True
) -> tf.data.Dataset:
    paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    def parse_and_preprocess(path_tensor, label_tensor):
        img = tf.numpy_function(func=read_image_from_path, inp=[path_tensor], Tout=tf.float32)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BICUBIC)

        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k=k)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

        # ImageNet standardization: (img / 255.0 - mean) / std
        mean = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=tf.float32) * 255.0
        std = tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=tf.float32) * 255.0
        img = (img - mean) / std

        return img, label_tensor

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    dataset = dataset.map(parse_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def create_tf_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None,
    subsample: Optional[int] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], int, Tuple[List[str], np.ndarray]]:
    active_seed = setup_random_seed(seed)
    image_paths, labels, class_names = load_histology_paths_and_labels(data_dir)

    if subsample is not None and subsample < len(image_paths):
        np.random.seed(active_seed)
        sub_idx = np.random.choice(len(image_paths), size=subsample, replace=False)
        image_paths = [image_paths[i] for i in sub_idx]
        labels = labels[sub_idx]

    holdout_ratio = val_split + test_split
    # Check minimum class count for stratify
    _, counts = np.unique(labels, return_counts=True)
    stratify_1 = labels if np.min(counts) >= 2 else None

    train_paths, holdout_paths, train_lbls, holdout_lbls = train_test_split(
        image_paths, labels, test_size=holdout_ratio, stratify=stratify_1, random_state=active_seed
    )

    test_rel_ratio = test_split / holdout_ratio
    _, holdout_counts = np.unique(holdout_lbls, return_counts=True)
    stratify_2 = holdout_lbls if np.min(holdout_counts) >= 2 else None

    val_paths, test_paths, val_lbls, test_lbls = train_test_split(
        holdout_paths, holdout_lbls, test_size=test_rel_ratio, stratify=stratify_2, random_state=active_seed
    )

    print(f"[Dataset] Split sizes: Train={len(train_paths)} (70%), Val={len(val_paths)} (15%), Test={len(test_paths)} (15%)")

    train_ds = build_tf_dataset(train_paths, train_lbls, img_size=img_size, batch_size=batch_size, is_training=True)
    val_ds = build_tf_dataset(val_paths, val_lbls, img_size=img_size, batch_size=batch_size, is_training=False)
    test_ds = build_tf_dataset(test_paths, test_lbls, img_size=img_size, batch_size=batch_size, is_training=False)

    return train_ds, val_ds, test_ds, class_names, active_seed, (test_paths, test_lbls)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_path = current_dir / "data"
    tr_ds, vl_ds, ts_ds, names, s, (t_paths, t_lbls) = create_tf_dataloaders(data_path, batch_size=16)
    for imgs, lbls in tr_ds.take(1):
        print(f"[Sanity Test] Batch tensor shape: {imgs.shape}, Labels: {lbls.shape}, Seed: {s}")

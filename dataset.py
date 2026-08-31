"""
Dataset loading and downloading utilities for the Kather 2016 Colorectal Histology dataset.
(5,000 histological image patches, 8 classes, 150x150 pixels, H&E stained).
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm


# Official Zenodo repository URL for Kather et al. (2016)
ZENODO_URL = "https://zenodo.org/records/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1"
MIRROR_URL = "https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip"

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
    "01_TUMOR": "Tumor epithelium (TUM)",
    "02_STROMA": "Simple stroma (STR)",
    "03_COMPLEX": "Complex stroma (CPX)",
    "04_LYMPHO": "Immune cells / Lymphocytes (LYM)",
    "05_DEBRIS": "Debris (DEB)",
    "06_MUCOSA": "Normal mucosal glands (NORM)",
    "07_ADIPOSE": "Adipose tissue (ADI)",
    "08_EMPTY": "Background / Empty (BACK)"
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract(data_dir: Path) -> Path:
    """
    Downloads the Kather 2016 dataset zip archive and extracts it into data_dir.
    Returns the path to the extracted image folder.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already extracted
    for candidate in [
        data_dir / "Kather_texture_2016_image_tiles_5000",
        data_dir,
    ]:
        if (candidate / "01_TUMOR").is_dir():
            print(f"[Dataset] Found existing dataset directory at: {candidate}")
            return candidate

    zip_path = data_dir / "Kather_texture_2016_image_tiles_5000.zip"
    
    if not zip_path.exists():
        print(f"[Dataset] Downloading Kather 2016 dataset from Zenodo (~57 MB)...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(ZENODO_URL, filename=zip_path, reporthook=t.update_to)
        except Exception as e:
            print(f"[Dataset] Download from primary URL failed ({e}), trying mirror...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(MIRROR_URL, filename=zip_path, reporthook=t.update_to)
    
    print(f"[Dataset] Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Verify directory structure
    for candidate in [
        data_dir / "Kather_texture_2016_image_tiles_5000",
        data_dir,
    ]:
        if (candidate / "01_TUMOR").is_dir():
            print(f"[Dataset] Extraction complete: {candidate}")
            return candidate
            
    raise FileNotFoundError(f"Could not locate class folders in {data_dir} after extraction.")


def load_dataset_paths_and_labels(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Scans the extracted dataset directory and returns:
    - image_paths: list of absolute file paths to all images
    - labels: 1D numpy array of integer class labels (0 to 7)
    - class_names: list of class names
    """
    data_dir = Path(data_dir)
    if not (data_dir / "01_TUMOR").is_dir():
        data_dir = download_and_extract(data_dir)

    image_paths: List[str] = []
    labels: List[int] = []

    valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = data_dir / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing expected class directory: {class_folder}")
        
        folder_files = [
            str(p.resolve()) for p in class_folder.iterdir()
            if p.suffix.lower() in valid_extensions
        ]
        
        if len(folder_files) == 0:
            raise ValueError(f"No images found in {class_folder}")
            
        image_paths.extend(folder_files)
        labels.extend([class_idx] * len(folder_files))

    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[Dataset] Successfully indexed {len(image_paths)} images across {len(CLASS_NAMES)} classes.")
    return image_paths, labels_arr, CLASS_NAMES


def load_single_image(image_path: str) -> np.ndarray:
    """
    Loads an image file and returns it as a uint8 RGB numpy array (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_from_tfds() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Fallback loader if user prefers loading via tensorflow_datasets.
    Returns (images_array, labels_array, class_names).
    """
    try:
        import tensorflow_datasets as tfds
        print("[Dataset] Loading 'colorectal_histology' from tensorflow_datasets...")
        ds, info = tfds.load('colorectal_histology', split='train', with_info=True, as_supervised=True)
        images, labels = [], []
        for img, lbl in tfds.as_numpy(ds):
            images.append(img)
            labels.append(lbl)
        return np.array(images), np.array(labels), info.features['label'].names
    except ImportError:
        raise ImportError("tensorflow_datasets is not installed. Use load_dataset_paths_and_labels instead.")


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    data_path = current_dir / "data"
    paths, labels, names = load_dataset_paths_and_labels(data_path)
    print(f"Sample path: {paths[0]}")
    print(f"Total samples: {len(paths)}, Labels shape: {labels.shape}")
    for idx, name in enumerate(names):
        count = np.sum(labels == idx)
        print(f"  Class {idx} ({name} - {CLASS_DESCRIPTIONS[name]}): {count} samples")

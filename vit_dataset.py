"""
PyTorch Dataset and Histology Data Augmentation Pipeline for Vision Transformers.
Tailored for H&E whole-slide histological image patches (colorectal_histology).
Includes D4 dihedral rotations (90°, 180°, 270°), flips, stain color perturbations,
and resolution interpolation for Vision Transformer architectures (EVA-02, ViT, Swin).
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from dataset import load_dataset_paths_and_labels, CLASS_NAMES, CLASS_DESCRIPTIONS


class RandomDihedralRotation(object):
    """
    Randomly applies one of the 8 transformations from the dihedral group D4:
    (0°, 90°, 180°, 270° rotations combined with horizontal/vertical flips).
    Reflects the true spatial rotational symmetry of histological tissue patches.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        # 0: None, 1: 90 deg, 2: 180 deg, 3: 270 deg
        rot_choice = random.randint(0, 3)
        if rot_choice == 1:
            img = img.transpose(Image.Transpose.ROTATE_90)
        elif rot_choice == 2:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif rot_choice == 3:
            img = img.transpose(Image.Transpose.ROTATE_270)
        
        # Horizontal / Vertical flip
        if random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            
        return img


class ColorectalHistologyDataset(Dataset):
    """
    PyTorch Dataset wrapper for colorectal histology image patches.
    """
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


def get_vit_transforms(
    img_size: int = 224,
    is_training: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> transforms.Compose:
    """
    Constructs histology data transforms with resolution scaling and domain augmentations.
    """
    if is_training:
        return transforms.Compose([
            # Bicubic interpolation is standard for Vision Transformers (e.g. ViT, EVA-02)
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            RandomDihedralRotation(),
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


def create_vit_dataloaders(
    data_dir: Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    subsample: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Loads dataset, performs stratified splitting (train/val/test), and returns DataLoaders.
    """
    image_paths, labels, class_names = load_dataset_paths_and_labels(data_dir)

    if subsample is not None and subsample < len(image_paths):
        print(f"[ViT-Dataset] Subsampling {subsample} images for fast training/testing...")
        np.random.seed(seed)
        sub_indices = np.random.choice(len(image_paths), size=subsample, replace=False)
        image_paths = [image_paths[i] for i in sub_indices]
        labels = labels[sub_indices]

    # Stratified Train / (Val + Test) Split
    holdout_ratio = val_split + test_split
    train_paths, holdout_paths, train_lbls, holdout_lbls = train_test_split(
        image_paths, labels, test_size=holdout_ratio, stratify=labels, random_state=seed
    )

    # Stratified Val / Test Split
    test_rel_ratio = test_split / holdout_ratio
    val_paths, test_paths, val_lbls, test_lbls = train_test_split(
        holdout_paths, holdout_lbls, test_size=test_rel_ratio, stratify=holdout_lbls, random_state=seed
    )

    print(f"[ViT-Dataset] Split sizes: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    train_tf = get_vit_transforms(img_size=img_size, is_training=True, mean=mean, std=std)
    eval_tf = get_vit_transforms(img_size=img_size, is_training=False, mean=mean, std=std)

    train_dataset = ColorectalHistologyDataset(train_paths, train_lbls, transform=train_tf)
    val_dataset = ColorectalHistologyDataset(val_paths, val_lbls, transform=eval_tf)
    test_dataset = ColorectalHistologyDataset(test_paths, test_lbls, transform=eval_tf)

    # Pin memory for fast GPU transfer if CUDA is available
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test transforms on dummy PIL image
    img = Image.new("RGB", (150, 150), color=(180, 50, 120))
    tf = get_vit_transforms(img_size=224, is_training=True)
    out_tensor = tf(img)
    print(f"[Sanity Test] ViT Transform output tensor shape: {out_tensor.shape}")

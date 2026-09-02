"""
Hugging Face Kvasir-SEG Dataset Loader & Segmentation Mask Pipeline.
Handles Pixel-Level Binary Semantic Segmentation from kowndinya23/Kvasir-SEG:
- image: RGB colonoscopy frame [H, W, 3]
- annotation: Binary ground truth mask [H, W] (255 = polyp lesion, 0 = background mucosa)

Protocol: 70% Train / 15% Val / 15% Test with dynamic randomized seeds.
"""

import os
import random
import secrets
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def setup_random_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = secrets.randbelow(900000) + 100000

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Random-Seed] Active experiment seed: {seed}")
    return seed


def generate_synthetic_demo_kvasir(output_dir: Path, num_samples: int = 50) -> List[Dict[str, Any]]:
    print(f"[Dataset] Generating {num_samples} synthetic Kvasir-SEG image-mask pairs in {output_dir}...")
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for idx in range(num_samples):
        w, h = 384, 384
        base_img = Image.new("RGB", (w, h), color=(
            random.randint(150, 200),
            random.randint(50, 90),
            random.randint(50, 80)
        ))
        mask_img = Image.new("L", (w, h), color=0)

        draw_img = ImageDraw.Draw(base_img)
        draw_mask = ImageDraw.Draw(mask_img)

        # Central lumen
        draw_img.ellipse([(140, 140), (244, 244)], fill=(35, 15, 20))

        # Add polyp bump and corresponding binary ground-truth mask
        has_polyp = random.random() < 0.90
        if has_polyp:
            px = random.randint(40, 260)
            py = random.randint(40, 260)
            pw = random.randint(50, 110)
            ph = random.randint(50, 110)

            # Draw lesion on image
            draw_img.ellipse([(px, py), (px + pw, py + ph)], fill=(220, 120, 110), outline=(130, 40, 40), width=3)
            for _ in range(6):
                vx = random.randint(px + 8, px + pw - 8)
                vy = random.randint(py + 8, py + ph - 8)
                draw_img.line([(vx, vy), (vx + 3, vy + 3)], fill=(170, 30, 30), width=1)

            # Draw exact binary mask (255 = Polyp)
            draw_mask.ellipse([(px, py), (px + pw, py + ph)], fill=255)

        base_img = base_img.filter(ImageFilter.GaussianBlur(radius=0.5))

        sample_name = f"kvasir_demo_{idx:04d}"
        img_file = img_dir / f"{sample_name}.jpg"
        mask_file = mask_dir / f"{sample_name}_mask.png"

        base_img.save(img_file)
        mask_img.save(mask_file)

        samples.append({
            "name": sample_name,
            "image_path": str(img_file),
            "mask_path": str(mask_file)
        })

    print(f"[Dataset] Generated {len(samples)} synthetic Kvasir-SEG image-mask pairs.")
    return samples


class KvasirSegmentationDataset(Dataset):
    def __init__(self, items: Any, target_size: Tuple[int, int] = (384, 384), is_training: bool = False):
        self.items = items
        self.target_size = target_size
        self.is_training = is_training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # 1. Load Image
        if "image_path" in item:
            image = Image.open(item["image_path"]).convert("RGB")
        elif "image" in item:
            img_val = item["image"]
            if isinstance(img_val, Image.Image):
                image = img_val.convert("RGB")
            elif isinstance(img_val, str):
                image = Image.open(img_val).convert("RGB")
            else:
                image = Image.fromarray(np.uint8(img_val)).convert("RGB")
        else:
            image = Image.new("RGB", self.target_size, color=(160, 60, 60))

        # 2. Load Annotation Mask
        if "mask_path" in item:
            mask = Image.open(item["mask_path"]).convert("L")
        elif "annotation" in item:
            ann_val = item["annotation"]
            if isinstance(ann_val, Image.Image):
                mask = ann_val.convert("L")
            elif isinstance(ann_val, str):
                mask = Image.open(ann_val).convert("L")
            else:
                mask = Image.fromarray(np.uint8(ann_val)).convert("L")
        else:
            mask = Image.new("L", self.target_size, color=0)

        # 3. Resize
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.target_size, Image.Resampling.NEAREST)

        # 4. Augmentations if training
        if self.is_training:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                rot = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
                image = image.transpose(rot)
                mask = mask.transpose(rot)

        # 5. Convert to Tensors
        img_np = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

        # Binary Mask tensor: [1, H, W] in {0.0, 1.0}
        mask_np = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        name = item.get("name", f"kvasir_{idx}")

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "pil_image": image,
            "pil_mask": mask,
            "name": str(name)
        }


def collate_seg_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    pil_images = [item["pil_image"] for item in batch]
    pil_masks = [item["pil_mask"] for item in batch]
    names = [item["name"] for item in batch]

    return {
        "image": images,
        "mask": masks,
        "pil_image": pil_images,
        "pil_mask": pil_masks,
        "name": names
    }


def create_kvasir_dataloaders(
    data_dir: Path,
    target_size: Tuple[int, int] = (384, 384),
    batch_size: int = 8,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: Optional[int] = None,
    subsample: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, int, Any]:
    active_seed = setup_random_seed(seed)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        print("[Dataset] Loading cached/remote kowndinya23/Kvasir-SEG from Hugging Face...")
        ds = load_dataset("kowndinya23/Kvasir-SEG", cache_dir=str(data_dir))
        
        available_splits = [ds[s] for s in ds.keys()]
        from datasets import concatenate_datasets
        full_ds = concatenate_datasets(available_splits) if len(available_splits) > 1 else available_splits[0]
        
        total_len = len(full_ds)
        print(f"[Dataset] Indexed {total_len} official Kvasir-SEG image-mask pairs from Hugging Face.")
        
        indices = list(range(total_len))
        if subsample is not None and subsample < total_len:
            np.random.seed(active_seed)
            indices = list(np.random.choice(total_len, size=subsample, replace=False))

        holdout_ratio = val_split + test_split
        train_idx, holdout_idx = train_test_split(indices, test_size=holdout_ratio, random_state=active_seed)
        test_rel_ratio = test_split / holdout_ratio
        val_idx, test_idx = train_test_split(holdout_idx, test_size=test_rel_ratio, random_state=active_seed)

        print(f"[Dataset] Strict Split: Train={len(train_idx)} (70%), Val={len(val_idx)} (15%), Test={len(test_idx)} (15%)")

        train_sub = full_ds.select(train_idx)
        val_sub = full_ds.select(val_idx)
        test_sub = full_ds.select(test_idx)

        tr_loader = DataLoader(KvasirSegmentationDataset(train_sub, target_size=target_size, is_training=True), batch_size=batch_size, shuffle=True, collate_fn=collate_seg_fn)
        vl_loader = DataLoader(KvasirSegmentationDataset(val_sub, target_size=target_size, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=collate_seg_fn)
        ts_loader = DataLoader(KvasirSegmentationDataset(test_sub, target_size=target_size, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=collate_seg_fn)

        return tr_loader, vl_loader, ts_loader, active_seed, test_sub

    except Exception as e:
        print(f"[Dataset] Direct Hugging Face access notice ({e}), loading demo dataset...")

    # Fallback / Local Demo Samples
    demo_samples = generate_synthetic_demo_kvasir(data_dir / "demo_kvasir", num_samples=50)
    holdout_ratio = val_split + test_split
    train_s, holdout_s = train_test_split(demo_samples, test_size=holdout_ratio, random_state=active_seed)
    test_rel_ratio = test_split / holdout_ratio
    val_s, test_s = train_test_split(holdout_s, test_size=test_rel_ratio, random_state=active_seed)

    print(f"[Dataset] Demo Split: Train={len(train_s)} (70%), Val={len(val_s)} (15%), Test={len(test_s)} (15%)")

    train_loader = DataLoader(KvasirSegmentationDataset(train_s, target_size=target_size, is_training=True), batch_size=batch_size, shuffle=True, collate_fn=collate_seg_fn)
    val_loader = DataLoader(KvasirSegmentationDataset(val_s, target_size=target_size, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=collate_seg_fn)
    test_loader = DataLoader(KvasirSegmentationDataset(test_s, target_size=target_size, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=collate_seg_fn)

    return train_loader, val_loader, test_loader, active_seed, test_s


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    tr_loader, val_loader, ts_loader, seed, test_samples = create_kvasir_dataloaders(current_dir / "data", batch_size=4)
    for batch in tr_loader:
        print(f"[Sanity Check] Batch image tensor: {batch['image'].shape}, Mask tensor: {batch['mask'].shape}")
        print(f"[Sanity Check] Mask value range: min={batch['mask'].min().item()}, max={batch['mask'].max().item()}")
        break

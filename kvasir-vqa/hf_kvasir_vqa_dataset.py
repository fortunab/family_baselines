"""
Hugging Face Kvasir-VQA Dataset Loader & Preprocessing Pipeline.
Handles Visual Question Answering on Gastrointestinal & Colorectal Endoscopy:
- image: RGB colonoscopy / endoscopy frame [H, W, 3]
- question: Clinical inquiry (e.g. 'is there a polyp present?', 'what tool is visible?')
- answer: Ground truth answer (closed-ended 'yes'/'no' or clinical terminology)
- img_id: Frame identifier
- source: Source dataset (HyperKvasir / Kvasir-Instrument)

Protocol: 70% Train / 15% Val / 15% Test with dynamic randomized seeds.
"""

import os
import re
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


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def is_closed_ended(answer: str) -> bool:
    cleaned = clean_text(answer)
    return cleaned in ["yes", "no"]


def generate_synthetic_demo_kvasir_vqa(output_dir: Path, num_samples: int = 50) -> List[Dict[str, Any]]:
    print(f"[Dataset] Generating {num_samples} synthetic Kvasir-VQA samples in {output_dir}...")
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    questions_bank = [
        ("Is there a polyp in this endoscopy image?", "yes", True),
        ("Is there an endoscopic instrument visible?", "no", True),
        ("Is the mucosa inflamed or showing erythema?", "no", True),
        ("Is there bleeding observed in the lumen?", "no", True),
        ("Where is the polyp located in the frame?", "center", False),
        ("What type of instrument is inserted?", "snare", False),
        ("Which anatomical landmark is visible?", "cecum", False),
        ("What endoscopic finding is shown?", "sessile polyp", False)
    ]

    samples = []
    for idx in range(num_samples):
        w, h = 384, 384
        # Synthetic colonoscopy frame (Red/Pink mucosal tones)
        img = Image.new("RGB", (w, h), color=(
            random.randint(160, 200),
            random.randint(50, 90),
            random.randint(50, 80)
        ))
        draw = ImageDraw.Draw(img)

        # Draw central dark lumen
        draw.ellipse([(140, 140), (244, 244)], fill=(35, 15, 20))

        # Draw lesion
        draw.ellipse([(100, 80), (180, 160)], fill=(210, 110, 100), outline=(120, 30, 30), width=2)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

        img_file = img_dir / f"kvasir_vqa_demo_{idx:04d}.jpg"
        img.save(img_file)

        q_item = random.choice(questions_bank)
        samples.append({
            "image_path": str(img_file),
            "question": q_item[0],
            "answer": q_item[1],
            "is_closed": q_item[2],
            "img_id": f"kvasir_vqa_demo_{idx:04d}",
            "source": "HyperKvasir"
        })

    print(f"[Dataset] Generated {len(samples)} synthetic Kvasir-VQA samples.")
    return samples


class KvasirVQADataset(Dataset):
    def __init__(self, items: Any, target_size: Tuple[int, int] = (384, 384)):
        self.items = items
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        # 1. Image loading
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

        image = image.resize(self.target_size, Image.Resampling.BILINEAR)

        # 2. Question, Answer, and Metadata
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        img_id = str(item.get("img_id", f"frame_{idx}"))
        source = str(item.get("source", "Kvasir-VQA"))

        # 3. Image Tensor (ImageNet standardized)
        img_np = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

        return {
            "image": img_tensor,
            "pil_image": image,
            "question": question,
            "answer": answer,
            "is_closed": is_closed_ended(answer),
            "img_id": img_id,
            "source": source
        }


def collate_vqa_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    pil_images = [item["pil_image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    is_closed = [item["is_closed"] for item in batch]
    img_ids = [item["img_id"] for item in batch]
    sources = [item["source"] for item in batch]

    return {
        "image": images,
        "pil_image": pil_images,
        "question": questions,
        "answer": answers,
        "is_closed": is_closed,
        "img_id": img_ids,
        "source": sources
    }


def create_kvasir_vqa_dataloaders(
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
        print("[Dataset] Loading cached/remote SimulaMet-HOST/Kvasir-VQA from Hugging Face...")
        ds = load_dataset("SimulaMet-HOST/Kvasir-VQA", cache_dir=str(data_dir))

        available_splits = [ds[s] for s in ds.keys()]
        from datasets import concatenate_datasets
        full_ds = concatenate_datasets(available_splits) if len(available_splits) > 1 else available_splits[0]

        total_len = len(full_ds)
        print(f"[Dataset] Indexed {total_len} official Kvasir-VQA pairs from Hugging Face.")

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

        tr_loader = DataLoader(KvasirVQADataset(train_sub, target_size=target_size), batch_size=batch_size, shuffle=True, collate_fn=collate_vqa_fn)
        vl_loader = DataLoader(KvasirVQADataset(val_sub, target_size=target_size), batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_fn)
        ts_loader = DataLoader(KvasirVQADataset(test_sub, target_size=target_size), batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_fn)

        return tr_loader, vl_loader, ts_loader, active_seed, test_sub

    except Exception as e:
        print(f"[Dataset] Direct Hugging Face access notice ({e}), loading demo dataset...")

    # Fallback / Local Demo Samples
    demo_samples = generate_synthetic_demo_kvasir_vqa(data_dir / "demo_kvasir_vqa", num_samples=50)
    holdout_ratio = val_split + test_split
    train_s, holdout_s = train_test_split(demo_samples, test_size=holdout_ratio, random_state=active_seed)
    test_rel_ratio = test_split / holdout_ratio
    val_s, test_s = train_test_split(holdout_s, test_size=test_rel_ratio, random_state=active_seed)

    print(f"[Dataset] Demo Split: Train={len(train_s)} (70%), Val={len(val_s)} (15%), Test={len(test_s)} (15%)")

    train_loader = DataLoader(KvasirVQADataset(train_s, target_size=target_size), batch_size=batch_size, shuffle=True, collate_fn=collate_vqa_fn)
    val_loader = DataLoader(KvasirVQADataset(val_s, target_size=target_size), batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_fn)
    test_loader = DataLoader(KvasirVQADataset(test_s, target_size=target_size), batch_size=batch_size, shuffle=False, collate_fn=collate_vqa_fn)

    return train_loader, val_loader, test_loader, active_seed, test_s


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    tr_loader, val_loader, ts_loader, seed, test_samples = create_kvasir_vqa_dataloaders(current_dir / "data", batch_size=4)
    for batch in tr_loader:
        print(f"[Sanity Check] Batch image tensor: {batch['image'].shape}")
        print(f"[Sanity Check] Question: '{batch['question'][0]}'")
        print(f"[Sanity Check] Answer: '{batch['answer'][0]}' (Closed-Ended: {batch['is_closed'][0]}, Source: {batch['source'][0]})")
        break

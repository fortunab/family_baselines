"""Dataset Discovery, Splitting, and Slicing Manager for Alzheimer MRI.

Supports 4 canonical classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented.
Provides stratified train, validation, and test splits for both ML tabular features and deep vision models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ALZHEIMER_CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented",
]

CLASS_DESCRIPTIONS = {
    "NonDemented": "No Cognitive Impairment (Healthy Aging Control)",
    "VeryMildDemented": "Very Mild Impairment (Prodromal Preclinical AD)",
    "MildDemented": "Mild Impairment (Early Stage AD)",
    "ModerateDemented": "Moderate Impairment (Advanced Degenerative AD)",
}


def normalize_class_name(name: str) -> Optional[str]:
    """Maps folder and filename strings to canonical 4-class Alzheimer stages."""
    clean = name.replace(" ", "").replace("_", "").lower()
    if "non" in clean or "noimpairment" in clean:
        return "NonDemented"
    if "verymild" in clean:
        return "VeryMildDemented"
    if "mild" in clean:
        return "MildDemented"
    if "moderate" in clean:
        return "ModerateDemented"
    return None


def discover_image_paths(data_dir: Union[str, Path]) -> pd.DataFrame:
    """Discovers all valid MRI images across class folders and returns a structured DataFrame."""
    p = Path(data_dir).resolve()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    records = []

    if p.exists() and p.is_dir():
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_exts:
                parent_cls = normalize_class_name(f.parent.name) or normalize_class_name(f.stem)
                if parent_cls:
                    records.append(
                        {
                            "image_path": str(f.resolve()),
                            "class_name": parent_cls,
                            "label": ALZHEIMER_CLASSES.index(parent_cls),
                            "description": CLASS_DESCRIPTIONS[parent_cls],
                        }
                    )

    df = pd.DataFrame(records)
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates stratified train, validation, and test splits."""
    if df.empty:
        raise ValueError("Cannot split empty dataset DataFrame.")

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    # Then split train and validation
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Returns frequency count per Alzheimer stage."""
    counts = {}
    for cls in ALZHEIMER_CLASSES:
        counts[cls] = int((df["class_name"] == cls).sum())
    return counts

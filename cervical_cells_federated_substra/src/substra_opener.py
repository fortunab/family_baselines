"""
Substra Dataset Opener Implementation for Cervical Cytology.
Conforms to Substra Asset Schema (substratools.Opener) for local organization node data loading.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    from substratools import Opener as BaseOpener
except ImportError:
    # Standalone simulation fallback
    class BaseOpener:
        """Fallback base opener class when substratools is absent."""


class SubstraCervicalOpener(BaseOpener):
    """
    Substra Opener asset for multi-class cervical cytology single-cell patches.
    Handles data ingestion, patient feature loading, and predictions serialization.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

    def fake_data(self, n_samples: int = 14) -> pd.DataFrame:
        """Generates synthetic DataFrame for Substra debug execution."""
        from src.dataset import CLASS_NAMES, CLASS_SHORT_NAMES

        records = []
        for i in range(n_samples):
            cls_idx = i % len(CLASS_NAMES)
            records.append(
                {
                    "filepath": "",
                    "label": CLASS_NAMES[cls_idx],
                    "label_idx": cls_idx,
                    "class_short": CLASS_SHORT_NAMES[cls_idx],
                }
            )
        return pd.DataFrame(records)

    def get_data(self, folders: str | Path | list[str] | pd.DataFrame) -> pd.DataFrame:
        """
        Loads the cervical cytology manifest or table for the Substra organization node.
        Accepts a DataFrame or folder path.
        """
        if isinstance(folders, pd.DataFrame):
            return folders.copy()

        folder_path = Path(folders)
        if folder_path.is_file() and folder_path.suffix == ".csv":
            return pd.read_csv(folder_path)

        from src.dataset import match_class_name

        records = []
        for img_path in folder_path.rglob("*.*"):
            fname = img_path.name.lower()
            if fname.endswith(("-d.bmp", "-cyt.bmp")) or "_mask" in fname:
                continue

            if img_path.suffix.lower() in [".bmp", ".png", ".jpg", ".jpeg", ".tif"]:
                cls = match_class_name(img_path.parent.name) or match_class_name(img_path.name)
                if cls is not None:
                    records.append(
                        {
                            "filepath": str(img_path.resolve()),
                            "label": cls,
                        }
                    )
        return pd.DataFrame(records)

    def get_predictions(self, file_path: str | Path) -> np.ndarray:
        """Reads predictions file produced by a Substra predict task."""
        return np.load(str(file_path))

    def save_predictions(self, predictions: np.ndarray, file_path: str | Path) -> None:
        """Saves predictions array inside Substra node environment."""
        np.save(str(file_path), predictions)

    def load_image_batch(
        self,
        df_slice: pd.DataFrame,
        transform: Any | None = None,
    ) -> tuple[list[Any], list[int]]:
        """Loads and transforms a batch of cervical cytology images."""
        from src.dataset import CLASS_TO_IDX

        images = []
        labels = []
        for _, row in df_slice.iterrows():
            try:
                img = Image.open(row["filepath"]).convert("RGB")
                if transform:
                    img = transform(img)
                images.append(img)
                lbl_name = row.get("label")
                labels.append(CLASS_TO_IDX.get(lbl_name, int(row.get("label_idx", 0))))
            except Exception:
                continue
        return images, labels

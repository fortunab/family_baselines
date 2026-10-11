"""Preprocess Alzheimer MRI Dataset with Clinical Grayscale Enhancement.

Executes:
1. ITU-R BT.601 single-channel luminance conversion.
2. 2D Gaussian spatial noise filtration.
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE).
4. Parenchymal skull-strip thresholding and dynamic range normalization.
5. Saves processed images to data/processed_grayscale/ and exports diagnostic comparison plot.
"""

import argparse
import logging
from pathlib import Path

from PIL import Image

from src.config import get_default_config, load_config
from src.dataset import ALZHEIMER_CLASSES, discover_image_paths
from src.preprocessing import batch_preprocess_dataset, preprocess_mri_slice
from src.visualizer import plot_preprocessing_breakdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GrayscalePreproc")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Alzheimer MRI scans with grayscale enhancement.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw MRI scans")
    parser.add_argument("--output-dir", type=str, default="data/processed_grayscale", help="Destination path")
    parser.add_argument("--image-size", type=int, default=128, help="Target image dimension (HxW)")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    img_size = (args.image_size, args.image_size)

    logger.info(f"Scanning raw MRI scans in {raw_dir}...")
    df_raw = discover_image_paths(raw_dir)
    logger.info(f"Discovered {len(df_raw)} scans across {df_raw['class_name'].nunique()} stages.")

    logger.info(f"Executing batch grayscale preprocessing pipeline -> {out_dir} (size={img_size})...")
    processed_count = batch_preprocess_dataset(raw_dir, out_dir, target_size=img_size)
    logger.info(f"Successfully preprocessed and saved {processed_count} grayscale slices.")

    # Generate visual comparison for a sample slice from each class
    results_dir = Path("results/preprocessing")
    results_dir.mkdir(parents=True, exist_ok=True)

    for cls in ALZHEIMER_CLASSES:
        cls_scans = df_raw[df_raw["class_name"] == cls]
        if not cls_scans.empty:
            sample_img = cls_scans.iloc[0]["image_path"]
            preproc_res = preprocess_mri_slice(sample_img, target_size=img_size)
            out_plot_path = results_dir / f"preproc_breakdown_{cls}.png"
            plot_preprocessing_breakdown(preproc_res, class_name=cls, output_path=out_plot_path)
            logger.info(f"Exported diagnostic 5-panel plot for {cls} to {out_plot_path}")

    logger.info("Grayscale preprocessing step successfully completed!")


if __name__ == "__main__":
    main()

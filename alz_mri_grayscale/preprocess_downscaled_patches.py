"""Batch Downscaling and Patch Grid Preprocessing for Alzheimer MRI.

Generates:
1. 32x32 pixel downscaled slices in data/processed_32x32/
2. 8x8 pixel downscaled slices in data/processed_8x8/
3. 6-panel Multi-Scale Patch Grid diagnostic figures in results/patches/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset import ALZHEIMER_CLASSES, discover_image_paths
from src.patch_engine import downscale_mri_slice
from src.preprocessing import preprocess_mri_slice
from src.visualizer import plot_multiscale_patch_grid_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DownscalePatches")


def parse_args():
    parser = argparse.ArgumentParser(description="Downscale MRI scans to 32x32 and 8x8 patches.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw or processed scans")
    parser.add_argument("--dir-32", type=str, default="data/processed_32x32", help="Destination for 32x32")
    parser.add_argument("--dir-8", type=str, default="data/processed_8x8", help="Destination for 8x8")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    dir_32 = Path(args.dir_32)
    dir_8 = Path(args.dir_8)

    dir_32.mkdir(parents=True, exist_ok=True)
    dir_8.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning images from {raw_dir}...")
    df = discover_image_paths(raw_dir)
    logger.info(f"Discovered {len(df)} images.")

    logger.info(f"Downscaling dataset to 32x32 ({dir_32}) and 8x8 ({dir_8})...")
    for _, row in df.iterrows():
        img_p = Path(row["image_path"])
        cls_name = row["class_name"]

        target_dir_32 = dir_32 / cls_name
        target_dir_8 = dir_8 / cls_name
        target_dir_32.mkdir(parents=True, exist_ok=True)
        target_dir_8.mkdir(parents=True, exist_ok=True)

        res = preprocess_mri_slice(img_p, target_size=(128, 128))
        gray = res["processed"]

        # 32x32 downscaling
        img_32 = downscale_mri_slice(gray, target_size=(32, 32))
        pil_32 = Image.fromarray(np.uint8(img_32 * 255.0), mode="L")
        pil_32.save(target_dir_32 / f"{img_p.stem}_32x32.png")

        # 8x8 downscaling
        img_8 = downscale_mri_slice(gray, target_size=(8, 8))
        pil_8 = Image.fromarray(np.uint8(img_8 * 255.0), mode="L")
        pil_8.save(target_dir_8 / f"{img_p.stem}_8x8.png")

    logger.info("Successfully generated 32x32 and 8x8 downscaled patch datasets.")

    # Render multi-scale patch comparison for each class
    patch_results_dir = Path("results/patches")
    patch_results_dir.mkdir(parents=True, exist_ok=True)

    for cls in ALZHEIMER_CLASSES:
        cls_df = df[df["class_name"] == cls]
        if not cls_df.empty:
            sample_p = cls_df.iloc[0]["image_path"]
            res = preprocess_mri_slice(sample_p, target_size=(128, 128))
            gray = res["processed"]
            img_32 = downscale_mri_slice(gray, target_size=(32, 32))
            img_8 = downscale_mri_slice(gray, target_size=(8, 8))

            out_plot = patch_results_dir / f"multiscale_patch_grid_{cls}.png"
            plot_multiscale_patch_grid_comparison(
                gray_128=gray,
                img_32=img_32,
                img_8=img_8,
                class_name=cls,
                output_path=out_plot,
            )
            logger.info(f"Saved 6-panel patch diagnostic plot for {cls} to {out_plot}")

    logger.info("Downscaling and patch extraction step completed!")


if __name__ == "__main__":
    main()

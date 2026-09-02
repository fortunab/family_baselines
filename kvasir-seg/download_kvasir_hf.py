"""
Automated Hugging Face Downloader & Cache Manager for kowndinya23/Kvasir-SEG.
"""

import os
import argparse
from pathlib import Path


def download_kvasir_hf(cache_dir: Path = None):
    print("\n" + "="*80)
    print("  HUGGING FACE DATASET DOWNLOADER: kowndinya23/Kvasir-SEG")
    print("="*80)

    try:
        from datasets import load_dataset
        print("[HF-Downloader] Attempting download via `datasets.load_dataset('kowndinya23/Kvasir-SEG')`...")
        ds = load_dataset("kowndinya23/Kvasir-SEG", cache_dir=str(cache_dir) if cache_dir else None)
        print(f"[HF-Downloader] Successfully loaded dataset with splits:")
        for split in ds.keys():
            print(f"  • Split '{split}': {len(ds[split])} samples")
        return ds
    except Exception as e:
        print(f"[HF-Downloader] Notice / Exception during direct download: {e}")
        print("[HF-Downloader] Dataset pipeline will operate with local cache or fallback demo generator.")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download kowndinya23/Kvasir-SEG from Hugging Face")
    parser.add_argument("--cache-dir", type=str, default="./data_hf", help="Cache directory")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    download_kvasir_hf(cache_path)

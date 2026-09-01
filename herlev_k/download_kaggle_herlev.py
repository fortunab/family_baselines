"""
Automated Python Downloader for the Herlev Cervical Cytology Dataset.
Supports multiple automated strategies:
1. 'kagglehub' (Official modern Kaggle Python library)
2. 'kaggle' Python API (with auto-extraction)
3. Direct Public Academic Mirror (no credentials required)
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
import urllib.request


def download_via_kagglehub(dest_dir: Path) -> bool:
    """Strategy 1: Using modern official kagglehub library."""
    try:
        import kagglehub
        print("[Downloader] Attempting download via 'kagglehub'...")
        downloaded_path = kagglehub.dataset_download("yuvrajsinhachowdhury/herlev-dataset")
        print(f"[Downloader] Downloaded to cache: {downloaded_path}")
        
        # Copy contents to destination data directory
        src_path = Path(downloaded_path)
        for item in src_path.iterdir():
            target = dest_dir / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        print(f"[Downloader] Dataset successfully placed into: {dest_dir.resolve()}")
        return True
    except Exception as e:
        print(f"[Downloader] kagglehub method returned: {e}")
        return False


def download_via_kaggle_api(dest_dir: Path) -> bool:
    """Strategy 2: Using standard Kaggle Python API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        print("[Downloader] Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        print("[Downloader] Downloading & unzipping 'yuvrajsinhachowdhury/herlev-dataset'...")
        api.dataset_download_files(
            "yuvrajsinhachowdhury/herlev-dataset",
            path=str(dest_dir),
            unzip=True,
            quiet=False
        )
        print(f"[Downloader] Dataset successfully extracted into: {dest_dir.resolve()}")
        return True
    except Exception as e:
        print(f"[Downloader] Kaggle API returned: {e}")
        return False


def download_via_direct_academic_mirror(dest_dir: Path) -> bool:
    """Strategy 3: Direct academic mirror download (No login / API token needed)."""
    mirror_url = "https://raw.githubusercontent.com/jakevdp/data/master/cervical_cancer_herlev.zip"
    alt_mirror = "https://data.mendeley.com/public-files/datasets/zddrjv5dvd/files/54054a88-51ec-436f-8700-1c0f2095368a/file_downloaded"
    
    zip_target = dest_dir / "herlev_mirror.zip"
    print(f"[Downloader] Downloading from academic mirror: {mirror_url}...")

    try:
        import requests
        from tqdm import tqdm

        response = requests.get(mirror_url, stream=True, timeout=30)
        if response.status_code != 200:
            print(f"[Downloader] Primary mirror returned status {response.status_code}, trying secondary mirror...")
            response = requests.get(alt_mirror, stream=True, timeout=30)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB

        with open(zip_target, 'wb') as file, tqdm(
            desc="Downloading Herlev Archive",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)

        print(f"[Downloader] Extracting {zip_target} into {dest_dir}...")
        with zipfile.ZipFile(zip_target, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)

        if zip_target.exists():
            zip_target.unlink()

        print("[Downloader] Extraction complete!")
        return True

    except Exception as e:
        print(f"[Downloader] Direct mirror download returned: {e}")
        return False


def main():
    dest_dir = Path(__file__).parent / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("      AUTOMATED HERLEV CERVICAL CYTOLOGY DATASET DOWNLOADER")
    print("="*80)
    print(f" Target Directory : {dest_dir.resolve()}")
    print("="*80)

    # 1. Try kagglehub
    if download_via_kagglehub(dest_dir):
        return

    # 2. Try kaggle API
    if download_via_kaggle_api(dest_dir):
        return

    # 3. Try direct academic mirror
    if download_via_direct_academic_mirror(dest_dir):
        return

    print("\n" + "!"*80)
    print(" MANUAL DOWNLOAD INSTRUCTIONS:")
    print(" 1. Visit: https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset")
    print(" 2. Click 'Download' (zip archive)")
    print(f" 3. Extract the class folders into: {dest_dir.resolve()}")
    print("!"*80 + "\n")


if __name__ == "__main__":
    main()

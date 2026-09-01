"""
Automated Downloader for the Original Herlev Pap Smear Database (MDE-Lab).
Website: https://mde-lab.aegean.gr/index.php/downloads/
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path


def download_mde_herlev(dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("      ORIGINAL MDE-LAB HERLEV PAP SMEAR DATABASE DOWNLOADER")
    print("="*80)
    print(f" Target Directory : {dest_dir.resolve()}")
    print(" Source Website   : https://mde-lab.aegean.gr/index.php/downloads/")
    print("="*80)

    # Academic Mirrors for automated extraction
    mirror_urls = [
        "https://data.mendeley.com/public-files/datasets/zddrjv5dvd/files/54054a88-51ec-436f-8700-1c0f2095368a/file_downloaded",
        "https://github.com/jakevdp/data/raw/master/cervical_cancer_herlev.zip"
    ]

    try:
        import requests
        from tqdm import tqdm

        for url in mirror_urls:
            print(f"[Downloader] Attempting download from: {url}...")
            try:
                response = requests.get(url, stream=True, timeout=25)
                if response.status_code == 200:
                    zip_target = dest_dir / "mde_herlev.zip"
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024

                    with open(zip_target, 'wb') as f, tqdm(
                        desc="Downloading Herlev Database",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024
                    ) as bar:
                        for chunk in response.iter_content(block_size):
                            size = f.write(chunk)
                            bar.update(size)

                    print(f"[Downloader] Extracting into {dest_dir}...")
                    with zipfile.ZipFile(zip_target, 'r') as zip_ref:
                        zip_ref.extractall(dest_dir)

                    if zip_target.exists():
                        zip_target.unlink()

                    print("[Downloader] Download & extraction completed successfully!")
                    return True
            except Exception as e:
                print(f"[Downloader] Mirror attempt failed: {e}")

    except ImportError:
        print("[Notice] 'requests' or 'tqdm' not found.")

    print("\n" + "!"*80)
    print(" MANUAL DOWNLOAD INSTRUCTIONS FROM MDE-LAB:")
    print(" 1. Visit: https://mde-lab.aegean.gr/index.php/downloads/")
    print(" 2. Under 'Databases', click 'Herlev Pap Smear Database' to download the zip archive.")
    print(f" 3. Extract all 7 folders into: {dest_dir.resolve()}")
    print("!"*80 + "\n")
    return False


if __name__ == "__main__":
    dest = Path(__file__).parent / "data"
    download_mde_herlev(dest)

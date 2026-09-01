"""
Automated Downloader for the CVC-Colon Databases (CVC-ClinicDB / CVC-ColonDB).
Website: https://pages.cvc.uab.es/CVC-Colon/index.php/databases/
"""

import os
import sys
import zipfile
from pathlib import Path


def download_cvc_colon(dest_dir: Path):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("        CVC-COLON DATASET DOWNLOADER (CVC-ClinicDB / CVC-ColonDB)")
    print("="*80)
    print(f" Target Directory : {dest_dir.resolve()}")
    print(" Source Website   : https://pages.cvc.uab.es/CVC-Colon/index.php/databases/")
    print("="*80)

    # Public repository / academic mirrors for CVC-ClinicDB
    mirror_urls = [
        "https://www.dropbox.com/s/p50hgjp9m0bvave/CVC-ClinicDB.zip?dl=1",
        "https://github.com/laurenz-ha/CVC-ClinicDB/archive/refs/heads/master.zip"
    ]

    try:
        import requests
        from tqdm import tqdm

        for url in mirror_urls:
            print(f"[Downloader] Attempting download from: {url}...")
            try:
                response = requests.get(url, stream=True, timeout=25)
                if response.status_code == 200:
                    zip_target = dest_dir / "cvc_clinicdb.zip"
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024

                    with open(zip_target, 'wb') as f, tqdm(
                        desc="Downloading CVC Archive",
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
    print(" MANUAL DOWNLOAD INSTRUCTIONS FROM CVC-COLON:")
    print(" 1. Visit: https://pages.cvc.uab.es/CVC-Colon/index.php/databases/")
    print(" 2. Download CVC-ClinicDB (612 frames) or CVC-ColonDB (300 frames).")
    print(f" 3. Extract the image folders into: {dest_dir.resolve()}")
    print("!"*80 + "\n")
    return False


if __name__ == "__main__":
    dest = Path(__file__).parent / "data"
    download_cvc_colon(dest)

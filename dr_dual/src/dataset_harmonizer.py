"""Dual-Database Harmonization Engine and Clinical Dataset Synthesizer.

Provides complete harmonization, synthesis, and deduplication for:
1. Diabetic Retinopathy Dual Database:
   - DB1: EyePACS / APTOS 2019 (Tele-ophthalmology screening clinics)
   - DB2: Messidor-2 / IDRiD (Hospital-based Topcon cameras)
   - 5-Grade ICDR Scale (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative) + Binary Referral
2. Dermatology Dual Database:
   - DB1: ISIC Dermoscopy Archive
   - DB2: HAM10000 (ViDIR / Queensland Collection)
   - 7 Diagnostic Classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC) with Lesion ID tracking
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class DualDatabaseSample:
    """Standardized schema for dual-database clinical image records."""

    image_id: str
    image_path: str
    database: str  # e.g., 'eyepacs_aptos' or 'messidor_idrid'
    patient_id: str
    split: str  # 'train', 'val', 'test'
    label_idx: int
    label_name: str
    is_referable: bool
    image_hash: str
    width: int = 256
    height: int = 256
    clinical_site: str = "site_a"
    acquisition_camera: str = "camera_standard"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualDatabaseSample":
        return cls(**d)


def compute_file_md5(file_path: Union[str, Path]) -> str:
    """Computes MD5 hash of an image file for strict deduplication."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =========================================================================
# 1. Retinal Fundus Synthesis for Diabetic Retinopathy
# =========================================================================


def _synthesize_retinal_fundus_image(
    grade: int,
    db_name: str,
    width: int = 256,
    height: int = 256,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Generates realistic retinal fundus photography with grade-specific diabetic retinopathy lesions."""
    if rng is None:
        rng = random.Random()

    # 1. Retinal Mask (Circular aperture)
    y_idx, x_idx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2.0, width / 2.0
    radius = (min(height, width) / 2.0) * 0.94
    dist_from_center = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    retina_mask = dist_from_center <= radius

    # Base Fundus Background Color (Orange-Red Choroid / Retinal Pigment Epithelium)
    # Different databases have characteristic illumination profiles:
    if db_name == "eyepacs_aptos":
        # EyePACS screening clinics: warmer, slightly more variable illumination
        r_base = rng.randint(180, 220)
        g_base = rng.randint(85, 120)
        b_base = rng.randint(25, 55)
    else:
        # Messidor hospital Topcon cameras: slightly higher contrast, deeper red
        r_base = rng.randint(195, 235)
        g_base = rng.randint(75, 110)
        b_base = rng.randint(20, 45)

    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = r_base
    img[:, :, 1] = g_base
    img[:, :, 2] = b_base

    # Retinal texture & pigment epithelium mottling
    noise = np.random.normal(0, 6, (height, width, 1)).astype(np.float32)
    img += noise

    # Illumination gradient (center brighter, retinal edge vignette)
    illumination = np.clip(1.0 - (dist_from_center / radius) * 0.35, 0.4, 1.0)
    img *= illumination[:, :, np.newaxis]

    # 2. Optic Disc (Bright yellowish-orange circle, nasal side)
    disc_side = -1 if rng.random() < 0.5 else 1  # Left or right eye
    disc_cx = int(cx + disc_side * radius * 0.48)
    disc_cy = int(cy + rng.randint(-15, 15))
    disc_radius = int(radius * 0.17)

    dist_disc = np.sqrt((x_idx - disc_cx) ** 2 + (y_idx - disc_cy) ** 2)
    disc_mask = dist_disc <= disc_radius

    # Optic cup inside disc
    cup_mask = dist_disc <= (disc_radius * 0.45)

    img[disc_mask, 0] = np.clip(img[disc_mask, 0] + 55, 0, 255)
    img[disc_mask, 1] = np.clip(img[disc_mask, 1] + 65, 0, 255)
    img[disc_mask, 2] = np.clip(img[disc_mask, 2] + 45, 0, 255)

    img[cup_mask, 0] = np.clip(img[cup_mask, 0] + 25, 0, 255)
    img[cup_mask, 1] = np.clip(img[cup_mask, 1] + 35, 0, 255)
    img[cup_mask, 2] = np.clip(img[cup_mask, 2] + 30, 0, 255)

    # 3. Fovea Centralis (Darker avascular zone at optical center)
    fovea_cx = int(cx - disc_side * radius * 0.20)
    fovea_cy = int(cy)
    dist_fovea = np.sqrt((x_idx - fovea_cx) ** 2 + (y_idx - fovea_cy) ** 2)
    fovea_mask = dist_fovea <= (radius * 0.16)
    img[fovea_mask, 0] = np.clip(img[fovea_mask, 0] - 25, 0, 255)
    img[fovea_mask, 1] = np.clip(img[fovea_mask, 1] - 30, 0, 255)
    img[fovea_mask, 2] = np.clip(img[fovea_mask, 2] - 15, 0, 255)

    # 4. Retinal Blood Vessels (Arterioles and Venules radiating from disc)
    # Venules are darker and thicker; arterioles are thinner and lighter
    num_branches = rng.randint(6, 10)
    for _ in range(num_branches):
        angle = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(radius * 0.5, radius * 0.85)
        # Curved path
        v_pts = []
        cur_x, cur_y = float(disc_cx), float(disc_cy)
        step_len = 6.0
        cur_ang = angle
        for _ in range(int(length / step_len)):
            cur_x += step_len * math.cos(cur_ang)
            cur_y += step_len * math.sin(cur_ang)
            cur_ang += rng.uniform(-0.15, 0.15)
            v_pts.append((cur_x, cur_y))

        # Render vessel on image
        for px, py in v_pts:
            if 0 <= int(py) < height and 0 <= int(px) < width:
                # 2-3 pixel thickness
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = int(py) + dy, int(px) + dx
                        if 0 <= ny < height and 0 <= nx < width and retina_mask[ny, nx]:
                            img[ny, nx, 0] = max(0, img[ny, nx, 0] - 45)
                            img[ny, nx, 1] = max(0, img[ny, nx, 1] - 55)
                            img[ny, nx, 2] = max(0, img[ny, nx, 2] - 20)

    # 5. Diabetic Retinopathy Pathology Embedding by Clinical ICDR Grade
    # Grade 0: No DR -> Zero lesions
    # Grade 1: Mild NPDR -> Microaneurysms only (small dark red dots, 2-6)
    # Grade 2: Moderate NPDR -> Multiple microaneurysms (8-16), dot-blot hemorrhages (3-8), hard exudates (bright yellow, 4-10)
    # Grade 3: Severe NPDR -> Extensive hemorrhages (>20 in 4 quadrants), venous beading, cotton-wool spots (fluffy white)
    # Grade 4: Proliferative DR -> Neovascularization networks, large preretinal/vitreous hemorrhages, fibrous proliferation

    if grade >= 1:
        # Microaneurysms (tiny dark red spots, 1-2 px radius)
        num_ma = rng.randint(3, 7) if grade == 1 else (rng.randint(10, 25) if grade == 2 else rng.randint(30, 60))
        for _ in range(num_ma):
            mx = rng.randint(int(cx - radius * 0.7), int(cx + radius * 0.7))
            my = rng.randint(int(cy - radius * 0.7), int(cy + radius * 0.7))
            if 0 <= my < height and 0 <= mx < width and retina_mask[my, mx]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= my + dy < height and 0 <= mx + dx < width:
                            img[my + dy, mx + dx, 0] = max(0, img[my + dy, mx + dx, 0] - 65)
                            img[my + dy, mx + dx, 1] = max(0, img[my + dy, mx + dx, 1] - 70)
                            img[my + dy, mx + dx, 2] = max(0, img[my + dy, mx + dx, 2] - 25)

    if grade >= 2:
        # Dot and Blot Hemorrhages (larger dark lesions, 3-5 px radius)
        num_hems = rng.randint(5, 12) if grade == 2 else (rng.randint(18, 35) if grade == 3 else rng.randint(25, 45))
        for _ in range(num_hems):
            hx = rng.randint(int(cx - radius * 0.75), int(cx + radius * 0.75))
            hy = rng.randint(int(cy - radius * 0.75), int(cy + radius * 0.75))
            hr = rng.randint(2, 4)
            for dy in range(-hr, hr + 1):
                for dx in range(-hr, hr + 1):
                    if dx * dx + dy * dy <= hr * hr:
                        ny, nx = hy + dy, hx + dx
                        if 0 <= ny < height and 0 <= nx < width and retina_mask[ny, nx]:
                            img[ny, nx, 0] = max(0, img[ny, nx, 0] - 80)
                            img[ny, nx, 1] = max(0, img[ny, nx, 1] - 85)
                            img[ny, nx, 2] = max(0, img[ny, nx, 2] - 30)

        # Hard Exudates (bright yellow lipid deposits, 2-5 px radius)
        num_exudates = (
            rng.randint(6, 14) if grade == 2 else (rng.randint(15, 30) if grade == 3 else rng.randint(20, 40))
        )
        for _ in range(num_exudates):
            ex = rng.randint(int(cx - radius * 0.65), int(cx + radius * 0.65))
            ey = rng.randint(int(cy - radius * 0.65), int(cy + radius * 0.65))
            er = rng.randint(2, 4)
            for dy in range(-er, er + 1):
                for dx in range(-er, er + 1):
                    if dx * dx + dy * dy <= er * er:
                        ny, nx = ey + dy, ex + dx
                        if 0 <= ny < height and 0 <= nx < width and retina_mask[ny, nx]:
                            img[ny, nx, 0] = min(255, img[ny, nx, 0] + 65)
                            img[ny, nx, 1] = min(255, img[ny, nx, 1] + 65)
                            img[ny, nx, 2] = max(0, img[ny, nx, 2] - 10)  # Yellow hue

    if grade >= 3:
        # Cotton Wool Spots (soft fluffy nerve fiber layer infarcts, cloudy grayish-white)
        num_cws = rng.randint(3, 7) if grade == 3 else rng.randint(6, 12)
        for _ in range(num_cws):
            cwx = rng.randint(int(cx - radius * 0.6), int(cx + radius * 0.6))
            cwy = rng.randint(int(cy - radius * 0.6), int(cy + radius * 0.6))
            cwr = rng.randint(5, 8)
            for dy in range(-cwr, cwr + 1):
                for dx in range(-cwr, cwr + 1):
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= cwr * cwr:
                        ny, nx = cwy + dy, cwx + dx
                        if 0 <= ny < height and 0 <= nx < width and retina_mask[ny, nx]:
                            factor = 1.0 - (dist_sq / (cwr * cwr))
                            img[ny, nx, 0] = min(255, img[ny, nx, 0] + 50 * factor)
                            img[ny, nx, 1] = min(255, img[ny, nx, 1] + 55 * factor)
                            img[ny, nx, 2] = min(255, img[ny, nx, 2] + 45 * factor)

    if grade == 4:
        # Proliferative DR: Neovascularization fronds (fine irregular tortuous vessel tangled nets)
        # + Large preretinal hemorrhage
        num_nv = rng.randint(2, 4)
        for _ in range(num_nv):
            nvx = rng.randint(int(cx - radius * 0.5), int(cx + radius * 0.5))
            nvy = rng.randint(int(cy - radius * 0.5), int(cy + radius * 0.5))
            for _ in range(25):
                fx = nvx + rng.randint(-12, 12)
                fy = nvy + rng.randint(-12, 12)
                if 0 <= fy < height and 0 <= fx < width and retina_mask[fy, fx]:
                    img[fy, fx, 0] = max(0, img[fy, fx, 0] - 70)
                    img[fy, fx, 1] = max(0, img[fy, fx, 1] - 75)
                    img[fy, fx, 2] = max(0, img[fy, fx, 2] - 30)

        # Boat-shaped preretinal hemorrhage
        ph_x = rng.randint(int(cx - radius * 0.4), int(cx + radius * 0.4))
        ph_y = rng.randint(int(cy - radius * 0.4), int(cy + radius * 0.4))
        for dy in range(-6, 7):
            for dx in range(-12, 13):
                if (dx * dx) / 144.0 + (dy * dy) / 36.0 <= 1.0:
                    ny, nx = ph_y + dy, ph_x + dx
                    if 0 <= ny < height and 0 <= nx < width and retina_mask[ny, nx]:
                        img[ny, nx, 0] = max(0, img[ny, nx, 0] - 95)
                        img[ny, nx, 1] = max(0, img[ny, nx, 1] - 95)
                        img[ny, nx, 2] = max(0, img[ny, nx, 2] - 35)

    # Black outside the circular retinal aperture
    img[~retina_mask] = 0

    return np.clip(img, 0, 255).astype(np.uint8)


# =========================================================================
# 2. Dermatology Synthesis for ISIC & HAM10000 Harmonization
# =========================================================================


def _synthesize_dermoscopy_image(
    class_name: str,
    db_name: str,
    width: int = 256,
    height: int = 256,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Generates synthetic dermoscopic skin lesion image matching ISIC/HAM10000 morphology."""
    if rng is None:
        rng = random.Random()

    # Skin background (Fitzpatrick skin type II-III)
    skin_r = rng.randint(210, 240)
    skin_g = rng.randint(175, 205)
    skin_b = rng.randint(155, 185)

    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 0] = skin_r
    img[:, :, 1] = skin_g
    img[:, :, 2] = skin_b

    # Skin micro-relief texture
    img += np.random.normal(0, 5, (height, width, 1)).astype(np.float32)

    # Circular vignette from dermatoscope lens
    y_idx, x_idx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2.0, width / 2.0
    r_lens = (min(height, width) / 2.0) * 0.95
    dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    vignette = np.clip(1.0 - (dist / r_lens) ** 4, 0.0, 1.0)

    # Lesion Center & Radius
    lr = rng.randint(int(width * 0.18), int(width * 0.32))

    # Pigmentation & Border based on diagnosis
    if class_name == "MEL":  # Melanoma: Asymmetry, Irregular border, Multiple colors (brown, black, blue-white veil)
        shift_r, shift_g, shift_b = -120, -110, -90
        irregularity = 0.28
    elif class_name == "BCC":  # Basal Cell Carcinoma: Translucent, arborizing telangiectasia, blue-gray globules
        shift_r, shift_g, shift_b = -60, -75, -55
        irregularity = 0.15
    elif class_name == "VASC":  # Vascular lesion: Lacunae, red-purple to blue-black
        shift_r, shift_g, shift_b = 20, -120, -70
        irregularity = 0.10
    else:  # Nevus (NV), BKL, AKIEC, DF
        shift_r, shift_g, shift_b = -80, -70, -50
        irregularity = 0.12

    angles = np.arctan2(y_idx - cy, x_idx - cx)
    r_boundary = lr * (1.0 + irregularity * np.cos(3 * angles) + 0.1 * np.sin(5 * angles))
    lesion_mask = dist <= r_boundary

    img[lesion_mask, 0] = np.clip(img[lesion_mask, 0] + shift_r, 0, 255)
    img[lesion_mask, 1] = np.clip(img[lesion_mask, 1] + shift_g, 0, 255)
    img[lesion_mask, 2] = np.clip(img[lesion_mask, 2] + shift_b, 0, 255)

    img *= vignette[:, :, np.newaxis]
    return np.clip(img, 0, 255).astype(np.uint8)


# =========================================================================
# 3. Master Dataset Construction & Cross-Database Deduplication
# =========================================================================


def build_diabetic_retinopathy_dual_db(
    output_root: Union[str, Path],
    samples_per_db: int = 250,
    seed: int = 42,
) -> Tuple[List[DualDatabaseSample], Dict[str, Any]]:
    """Builds and harmonizes the Diabetic Retinopathy Dual-Database benchmark (EyePACS/APTOS + Messidor/IDRiD)."""
    rng = random.Random(seed)
    np.random.seed(seed)
    root = Path(output_root)

    db_dir = root / "data" / "raw" / "diabetic_retinopathy"
    db1_dir = db_dir / "eyepacs_aptos"
    db2_dir = db_dir / "messidor_idrid"

    db1_dir.mkdir(parents=True, exist_ok=True)
    db2_dir.mkdir(parents=True, exist_ok=True)

    dr_classes = ["No_DR", "Mild_NPDR", "Moderate_NPDR", "Severe_NPDR", "Proliferative_DR"]
    samples_per_class = samples_per_db // len(dr_classes)

    samples: List[DualDatabaseSample] = []
    seen_hashes = set()
    hash_collisions = 0

    # 1. Database 1: EyePACS / APTOS (Screening Clinics)
    print(f"Generating DB1: EyePACS / APTOS ({samples_per_db} scans)...")
    for grade_idx, grade_name in enumerate(dr_classes):
        for s_i in range(samples_per_class):
            img_id = f"eyepacs_{grade_name}_{s_i:04d}"
            pat_id = f"pat_ep_{grade_idx:02d}_{s_i:03d}"
            # Split assignment: 70% train, 15% val, 15% test (patient-level)
            split = (
                "train"
                if s_i < int(samples_per_class * 0.70)
                else ("val" if s_i < int(samples_per_class * 0.85) else "test")
            )

            img_arr = _synthesize_retinal_fundus_image(grade=grade_idx, db_name="eyepacs_aptos", rng=rng)
            img_path = db1_dir / f"{img_id}.png"
            Image.fromarray(img_arr).save(img_path)

            img_hash = compute_file_md5(img_path)
            if img_hash in seen_hashes:
                hash_collisions += 1
            seen_hashes.add(img_hash)

            samples.append(
                DualDatabaseSample(
                    image_id=img_id,
                    image_path=str(img_path.resolve()),
                    database="eyepacs_aptos",
                    patient_id=pat_id,
                    split=split,
                    label_idx=grade_idx,
                    label_name=grade_name,
                    is_referable=(grade_idx >= 2),
                    image_hash=img_hash,
                    clinical_site="tele_screening_clinic",
                    acquisition_camera="eyepacs_portable_nm",
                )
            )

    # 2. Database 2: Messidor / IDRiD (Hospital Topcon Cameras)
    print(f"Generating DB2: Messidor / IDRiD ({samples_per_db} scans)...")
    for grade_idx, grade_name in enumerate(dr_classes):
        for s_i in range(samples_per_class):
            img_id = f"messidor_{grade_name}_{s_i:04d}"
            pat_id = f"pat_mes_{grade_idx:02d}_{s_i:03d}"
            split = (
                "train"
                if s_i < int(samples_per_class * 0.70)
                else ("val" if s_i < int(samples_per_class * 0.85) else "test")
            )

            img_arr = _synthesize_retinal_fundus_image(grade=grade_idx, db_name="messidor_idrid", rng=rng)
            img_path = db2_dir / f"{img_id}.png"
            Image.fromarray(img_arr).save(img_path)

            img_hash = compute_file_md5(img_path)
            if img_hash in seen_hashes:
                hash_collisions += 1
            seen_hashes.add(img_hash)

            samples.append(
                DualDatabaseSample(
                    image_id=img_id,
                    image_path=str(img_path.resolve()),
                    database="messidor_idrid",
                    patient_id=pat_id,
                    split=split,
                    label_idx=grade_idx,
                    label_name=grade_name,
                    is_referable=(grade_idx >= 2),
                    image_hash=img_hash,
                    clinical_site="university_hospital_ophthalmology",
                    acquisition_camera="topcon_trc_nw6",
                )
            )

    # Save Manifest
    manifest_path = root / "data" / "manifests" / "dr_dual_db_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "benchmark_name": "Diabetic Retinopathy Dual-Database Harmonization Benchmark",
        "databases": ["eyepacs_aptos", "messidor_idrid"],
        "icdr_classes": dr_classes,
        "total_samples": len(samples),
        "hash_collisions": hash_collisions,
        "is_leakage_free": (hash_collisions == 0),
        "samples": [s.to_dict() for s in samples],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    audit_report = {
        "total_samples": len(samples),
        "db1_samples": sum(1 for s in samples if s.database == "eyepacs_aptos"),
        "db2_samples": sum(1 for s in samples if s.database == "messidor_idrid"),
        "train_samples": sum(1 for s in samples if s.split == "train"),
        "val_samples": sum(1 for s in samples if s.split == "val"),
        "test_samples": sum(1 for s in samples if s.split == "test"),
        "hash_collisions": hash_collisions,
        "is_leakage_free": (hash_collisions == 0),
        "manifest_path": str(manifest_path),
    }

    return samples, audit_report


def build_isic_ham10000_dual_db(
    output_root: Union[str, Path],
    samples_per_db: int = 150,
    seed: int = 42,
) -> Tuple[List[DualDatabaseSample], Dict[str, Any]]:
    """Builds and harmonizes the Dermatology Dual-Database benchmark (ISIC Archive + HAM10000)."""
    rng = random.Random(seed)
    np.random.seed(seed)
    root = Path(output_root)

    db_dir = root / "data" / "raw" / "dermatology_isic_ham"
    db1_dir = db_dir / "isic_archive"
    db2_dir = db_dir / "ham10000"

    db1_dir.mkdir(parents=True, exist_ok=True)
    db2_dir.mkdir(parents=True, exist_ok=True)

    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    samples_per_class = samples_per_db // len(classes)

    samples: List[DualDatabaseSample] = []
    seen_hashes = set()
    hash_collisions = 0

    for db_name, out_dir in [("isic_archive", db1_dir), ("ham10000", db2_dir)]:
        print(f"Generating {db_name} dermoscopy subset...")
        for c_idx, c_name in enumerate(classes):
            for s_i in range(samples_per_class):
                img_id = f"{db_name}_{c_name}_{s_i:04d}"
                pat_id = f"pat_{db_name[:4]}_{c_idx:02d}_{s_i:03d}"
                split = (
                    "train"
                    if s_i < int(samples_per_class * 0.70)
                    else ("val" if s_i < int(samples_per_class * 0.85) else "test")
                )

                img_arr = _synthesize_dermoscopy_image(class_name=c_name, db_name=db_name, rng=rng)
                img_path = out_dir / f"{img_id}.png"
                Image.fromarray(img_arr).save(img_path)

                img_hash = compute_file_md5(img_path)
                if img_hash in seen_hashes:
                    hash_collisions += 1
                seen_hashes.add(img_hash)

                samples.append(
                    DualDatabaseSample(
                        image_id=img_id,
                        image_path=str(img_path.resolve()),
                        database=db_name,
                        patient_id=pat_id,
                        split=split,
                        label_idx=c_idx,
                        label_name=c_name,
                        is_referable=(c_name in ["MEL", "BCC"]),
                        image_hash=img_hash,
                        clinical_site="dermatology_clinic",
                        acquisition_camera="dermatoscope_polarized",
                    )
                )

    manifest_path = root / "data" / "manifests" / "isic_ham_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "benchmark_name": "ISIC Archive + HAM10000 Dual-Database Harmonization Benchmark",
        "databases": ["isic_archive", "ham10000"],
        "dermatology_classes": classes,
        "total_samples": len(samples),
        "hash_collisions": hash_collisions,
        "is_leakage_free": (hash_collisions == 0),
        "samples": [s.to_dict() for s in samples],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    audit_report = {
        "total_samples": len(samples),
        "db1_samples": sum(1 for s in samples if s.database == "isic_archive"),
        "db2_samples": sum(1 for s in samples if s.database == "ham10000"),
        "train_samples": sum(1 for s in samples if s.split == "train"),
        "val_samples": sum(1 for s in samples if s.split == "val"),
        "test_samples": sum(1 for s in samples if s.split == "test"),
        "hash_collisions": hash_collisions,
        "is_leakage_free": (hash_collisions == 0),
        "manifest_path": str(manifest_path),
    }

    return samples, audit_report

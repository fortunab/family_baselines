"""CLI Tool: Generate and Harmonize Dual Databases (Diabetic Retinopathy & ISIC/HAM10000)."""

import argparse
import logging
from pathlib import Path

from src.dataset_harmonizer import (
    build_diabetic_retinopathy_dual_db,
    build_isic_ham10000_dual_db,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GenerateDualDatabases")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dual-Database Medical Benchmarks")
    parser.add_argument("--output-dir", type=str, default=".", help="Project root directory")
    parser.add_argument(
        "--dr-samples-per-db", type=int, default=250, help="Diabetic Retinopathy scans per DB (total = 2x)"
    )
    parser.add_argument("--isic-samples-per-db", type=int, default=150, help="ISIC/HAM10000 images per DB (total = 2x)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_dir)

    # 1. Build Diabetic Retinopathy Dual Database (EyePACS/APTOS + Messidor/IDRiD)
    logger.info("==================================================================")
    logger.info("Constructing Diabetic Retinopathy Dual-Database Benchmark...")
    logger.info("DB1: EyePACS / APTOS (Screening Clinics)")
    logger.info("DB2: Messidor-2 / IDRiD (Hospital Topcon Systems)")
    logger.info("==================================================================")
    dr_samples, dr_audit = build_diabetic_retinopathy_dual_db(
        output_root=root,
        samples_per_db=args.dr_samples_per_db,
        seed=args.seed,
    )
    logger.info(
        f"DR Total Scans: {dr_audit['total_samples']} | DB1: {dr_audit['db1_samples']} | DB2: {dr_audit['db2_samples']}"
    )
    logger.info(f"DR Deduplication Status: {'PASSED (Zero collisions)' if dr_audit['is_leakage_free'] else 'FAILED'}")

    # 2. Build Dermatology Dual Database (ISIC Archive + HAM10000)
    logger.info("==================================================================")
    logger.info("Constructing Dermatology Dual-Database Benchmark...")
    logger.info("DB1: ISIC Dermoscopy Archive")
    logger.info("DB2: HAM10000 ViDIR / Queensland Collection")
    logger.info("==================================================================")
    isic_samples, isic_audit = build_isic_ham10000_dual_db(
        output_root=root,
        samples_per_db=args.isic_samples_per_db,
        seed=args.seed,
    )
    logger.info(
        f"ISIC/HAM Total Images: {isic_audit['total_samples']} | DB1: {isic_audit['db1_samples']} | DB2: {isic_audit['db2_samples']}"
    )
    logger.info(
        f"ISIC/HAM Deduplication Status: {'PASSED (Zero collisions)' if isic_audit['is_leakage_free'] else 'FAILED'}"
    )

    logger.info("==================================================================")
    logger.info("All dual-database benchmarks synthesized and harmonized successfully.")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()

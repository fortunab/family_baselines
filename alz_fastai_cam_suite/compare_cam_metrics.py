"""Quantitative CAM Explainability Benchmark across Alzheimer MRI Scans.

Evaluates and compares:
- Grad-CAM (1st order gradient)
- Grad-CAM++ (Higher order gradient)
- Score-CAM (Perturbation forward-pass)

Outputs:
- Energy Localization Ratio (Brain vs Background)
- Average Drop (%) on Masking
- Drop (%) on Ablation
- Spatial Shannon Entropy
- Inter-CAM Structural Similarity (SSIM)
- Pearson Correlation Matrix
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.cam import CAMOrchestrator
from src.config import get_default_config, load_config
from src.dataset import build_dataloaders, discover_alzheimer_images, preprocess_image_for_eval
from src.evaluator import evaluate_cam_suite
from src.fastai_engine import build_fastai_learner, train_learner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlzheimerCAMBenchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CAM methods on Alzheimer MRI scans.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--samples-per-class", type=int, default=5, help="Number of MRI slices to test per class")
    parser.add_argument("--output", type=str, default="results/metrics/cam_benchmark_comparison.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config) if Path(args.config).exists() else get_default_config()

    data_dir = config["dataset"]["data_dir"]
    device = config["model"]["device"]
    image_size = config["dataset"]["image_size"]
    arch_str = config["model"]["arch"]

    logger.info("Initializing dataset and model for CAM benchmark...")
    dls, class_names = build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=config["dataset"]["batch_size"],
    )

    learner = build_fastai_learner(dls=dls, arch_str=arch_str, device=device)
    ckpt = args.checkpoint or config["training"]["model_save_path"]
    if Path(ckpt).exists():
        logger.info(f"Loading weights from {ckpt}...")
        learner.model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        logger.info("Training fastai model for 2 epochs for benchmark...")
        learner = train_learner(learner, epochs=2, lr=0.001)

    learner.model.eval()
    orchestrator = CAMOrchestrator(fastai_learner=learner, scorecam_max_channels=32, device=device)

    class_images = discover_alzheimer_images(data_dir)
    records = {"gradcam": [], "gradcam_plus_plus": [], "scorecam": []}
    ssim_records = {"gradcam_vs_gradcam_plus_plus": [], "gradcam_vs_scorecam": [], "gradcam_plus_plus_vs_scorecam": []}

    total_slices = 0
    for c_idx, c_name in enumerate(class_names):
        imgs = class_images.get(c_name, [])
        sampled = imgs[: args.samples_per_class]
        logger.info(f"Evaluating {len(sampled)} slices for class {c_name}...")

        for s_path in sampled:
            total_slices += 1
            tensor_img, raw_img_np = preprocess_image_for_eval(s_path, image_size=image_size)
            cams, _ = orchestrator.generate_all(
                input_tensor=tensor_img,
                raw_image_np=raw_img_np,
                target_class=c_idx,
            )

            res = evaluate_cam_suite(
                model=learner.model,
                tensor_img=tensor_img,
                cam_maps=cams,
                target_class=c_idx,
                raw_img_np=raw_img_np,
                device=device,
            )

            for m_name in records:
                if m_name in res["methods"]:
                    records[m_name].append(res["methods"][m_name])

            for pair_name in ssim_records:
                if pair_name in res["inter_cam_consistency"]:
                    ssim_records[pair_name].append(res["inter_cam_consistency"][pair_name])

    # Compute aggregate stats
    summary = {}
    for m_name, item_list in records.items():
        if not item_list:
            continue
        summary[m_name] = {
            "energy_localization_ratio_mean": round(
                float(np.mean([x["energy_localization_ratio"] for x in item_list])), 4
            ),
            "energy_localization_ratio_std": round(
                float(np.std([x["energy_localization_ratio"] for x in item_list])), 4
            ),
            "average_drop_pct_mean": round(float(np.mean([x["average_drop_pct"] for x in item_list])), 2),
            "average_drop_pct_std": round(float(np.std([x["average_drop_pct"] for x in item_list])), 2),
            "ablation_drop_pct_mean": round(float(np.mean([x["ablation_drop_pct"] for x in item_list])), 2),
            "ablation_drop_pct_std": round(float(np.std([x["ablation_drop_pct"] for x in item_list])), 2),
            "shannon_entropy_mean": round(float(np.mean([x["shannon_entropy"] for x in item_list])), 4),
            "shannon_entropy_std": round(float(np.std([x["shannon_entropy"] for x in item_list])), 4),
            "sparsity_mean": round(float(np.mean([x["sparsity"] for x in item_list])), 4),
        }

    ssim_summary = {}
    for pair_name, item_list in ssim_records.items():
        if not item_list:
            continue
        ssim_summary[pair_name] = {
            "ssim_mean": round(float(np.mean([x["ssim"] for x in item_list])), 4),
            "ssim_std": round(float(np.std([x["ssim"] for x in item_list])), 4),
            "pearson_mean": round(float(np.mean([x["pearson_corr"] for x in item_list])), 4),
            "pearson_std": round(float(np.std([x["pearson_corr"] for x in item_list])), 4),
        }

    output_data = {
        "slices_evaluated": total_slices,
        "classes": class_names,
        "method_benchmarks": summary,
        "inter_cam_consensus": ssim_summary,
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved CAM quantitative benchmark report to {out_file}")

    print("\n" + "=" * 80)
    print("QUANTITATIVE CAM EXPLAINABILITY BENCHMARK REPORT")
    print("=" * 80)
    print(f"{'Method':<20} | {'Brain Energy %':<18} | {'Mask Drop %':<18} | {'Ablation Drop %':<18} | {'Entropy'}")
    print("-" * 80)
    for m, vals in summary.items():
        energy_str = (
            f"{vals['energy_localization_ratio_mean']*100:.1f} ± {vals['energy_localization_ratio_std']*100:.1f}%"
        )
        drop_str = f"{vals['average_drop_pct_mean']:.1f} ± {vals['average_drop_pct_std']:.1f}%"
        abl_str = f"{vals['ablation_drop_pct_mean']:.1f} ± {vals['ablation_drop_pct_std']:.1f}%"
        ent_str = f"{vals['shannon_entropy_mean']:.2f} ± {vals['shannon_entropy_std']:.2f}"
        print(f"{m:<20} | {energy_str:<18} | {drop_str:<18} | {abl_str:<18} | {ent_str}")
    print("=" * 80)
    print("INTER-METHOD CONSISTENCY (SSIM & PEARSON CORRELATION):")
    for pair, pvals in ssim_summary.items():
        s_m, s_s = pvals["ssim_mean"], pvals["ssim_std"]
        p_m, p_s = pvals["pearson_mean"], pvals["pearson_std"]
        print(f"  - {pair}: SSIM = {s_m:.3f} ± {s_s:.3f} | Pearson = {p_m:.3f} ± {p_s:.3f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

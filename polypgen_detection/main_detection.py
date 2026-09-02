"""
Main CLI Entrypoint for COCO Object Detection on Hugging Face PolypGen2.0.
Supports 4 Foundation Detectors:
1. florence2 (Microsoft Florence-2-base / large)
2. owlv2 (Google OWLv2-base-patch16)
3. grounding_dino (Grounding DINO base / tiny)
4. paligemma (Google DeepMind PaliGemma 3B)
5. all (runs all 4 sequentially)
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch

from hf_polyp_dataset import create_polypgen_dataloaders, setup_random_seed
from coco_evaluator import evaluate_coco_detections, print_detection_report, plot_detection_overlays
from detector_florence2 import Florence2Detector
from detector_owlv2 import OWLv2Detector
from detector_grounding_dino import GroundingDINODetector
from detector_paligemma import PaliGemmaDetector


def parse_args():
    parser = argparse.ArgumentParser(description="COCO Foundation Object Detection on PolypGen2.0")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    parser.add_argument(
        "--model-name", type=str, default="florence2",
        choices=["florence2", "owlv2", "grounding_dino", "paligemma", "all"],
        help="Foundation detection model to evaluate"
    )
    parser.add_argument("--seed", type=int, default=None, help="Dynamic experiment seed")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast benchmarking")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    return parser.parse_args()


def get_detector(model_name: str):
    name_lower = model_name.lower()
    if "florence" in name_lower:
        return Florence2Detector(), "florence2"
    elif "owl" in name_lower:
        return OWLv2Detector(), "owlv2"
    elif "grounding" in name_lower or "dino" in name_lower:
        return GroundingDINODetector(), "grounding_dino"
    elif "pali" in name_lower or "gemma" in name_lower:
        return PaliGemmaDetector(), "paligemma"
    else:
        print(f"[Warning] Unknown model '{model_name}', defaulting to Florence-2...")
        return Florence2Detector(), "florence2"


def run_detection_benchmark(model_key: str, args, project_root: Path, active_seed: int):
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir

    print("\n" + "="*95)
    print(f"  STARTING FOUNDATION DETECTION BENCHMARK: {model_key.upper()} (POLYPGEN2.0)")
    print("="*95)

    detector, canonical_name = get_detector(model_key)

    train_loader, val_loader, test_loader, seed, test_samples = create_polypgen_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        seed=active_seed,
        subsample=args.subsample
    )

    print(f"[Benchmark] Evaluating on {len(test_loader.dataset)} unseen holdout test colonoscopy frames...")
    predictions = []
    ground_truths = []
    first_visual_sample = None

    for batch in tqdm(test_loader, desc=f"Evaluating {canonical_name}"):
        pil_images = batch["pil_images"]
        gt_boxes_batch = batch["boxes"]
        img_ids = batch["image_ids"]
        center_ids = batch["center_ids"]

        for pil_img, gt_boxes, img_id, c_id in zip(pil_images, gt_boxes_batch, img_ids, center_ids):
            det_res = detector.detect(pil_img)
            pred_boxes = det_res.get("boxes", [])
            pred_scores = det_res.get("scores", [])

            gt_boxes_list = gt_boxes.cpu().numpy().tolist() if torch.is_tensor(gt_boxes) else gt_boxes

            predictions.append({
                "image_id": img_id,
                "boxes": pred_boxes,
                "scores": pred_scores,
                "center_id": c_id
            })
            ground_truths.append({
                "image_id": img_id,
                "boxes": gt_boxes_list,
                "center_id": c_id
            })

            if first_visual_sample is None and len(gt_boxes_list) > 0:
                first_visual_sample = (pil_img, gt_boxes_list, pred_boxes)

    # Compute COCO Evaluation Metrics
    metrics = evaluate_coco_detections(predictions, ground_truths)
    metrics["Experiment_Seed"] = int(seed)
    metrics["Model_Name"] = canonical_name

    print_detection_report(metrics, canonical_name, seed=seed)

    # Save metrics JSON
    json_path = results_dir / f"metrics_summary_{canonical_name}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Main] Metrics saved to: {json_path}")

    # Save visual overlay
    if first_visual_sample is not None:
        vis_img, vis_gt, vis_pred = first_visual_sample
        overlay_path = results_dir / f"detection_overlay_{canonical_name}.png"
        plot_detection_overlays(
            vis_img, vis_gt, vis_pred,
            output_path=overlay_path,
            title=f"{canonical_name.upper()} Polyp Detection: GT (Green) vs. Pred (Red)"
        )

    return metrics


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    active_seed = setup_random_seed(args.seed)

    if args.model_name == "all":
        models = ["florence2", "owlv2", "grounding_dino", "paligemma"]
        print(f"[Main] Running full detection benchmark across all {len(models)} Foundation Models...")
        for m in models:
            run_detection_benchmark(m, args, project_root, active_seed)
    else:
        run_detection_benchmark(args.model_name, args, project_root, active_seed)


if __name__ == "__main__":
    main()

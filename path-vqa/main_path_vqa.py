"""
Main CLI Entrypoint for Pathology Visual Question Answering (PathVQA) Foundation Benchmark.
Supports 4 Foundation Models:
1. biomedclip (Microsoft BiomedCLIP / Florence-2)
2. paligemma (Google DeepMind PaliGemma)
3. qwen2vl (Alibaba Qwen2-VL)
4. blip2 (Salesforce BLIP-2)
5. all (runs all 4 sequentially)
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch

from hf_path_vqa_dataset import create_path_vqa_dataloaders, setup_random_seed
from vqa_metrics import evaluate_vqa_predictions, print_vqa_report, plot_vqa_diagnostic_card
from model_biomedclip import BiomedCLIPVQAModel
from model_paligemma_vqa import PaliGemmaVQAModel
from model_qwen2vl_vqa import Qwen2VLVQAModel
from model_blip2_vqa import BLIP2VQAModel


def parse_args():
    parser = argparse.ArgumentParser(description="Pathology Visual Question Answering (PathVQA) Benchmark")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    parser.add_argument(
        "--model-name", type=str, default="biomedclip",
        choices=["biomedclip", "paligemma", "qwen2vl", "blip2", "all"],
        help="VQA foundation model to evaluate"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seed", type=int, default=None, help="Dynamic experiment seed")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast testing")
    return parser.parse_args()


def get_vqa_model(model_name: str):
    name_lower = model_name.lower()
    if "biomed" in name_lower or "florence" in name_lower:
        return BiomedCLIPVQAModel(), "biomedclip"
    elif "paligemma" in name_lower:
        return PaliGemmaVQAModel(), "paligemma"
    elif "qwen" in name_lower:
        return Qwen2VLVQAModel(), "qwen2vl"
    elif "blip" in name_lower:
        return BLIP2VQAModel(), "blip2"
    else:
        print(f"[Warning] Unknown model '{model_name}', defaulting to BiomedCLIP...")
        return BiomedCLIPVQAModel(), "biomedclip"


def run_vqa_pipeline(model_key: str, args, project_root: Path, active_seed: int):
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir

    print("\n" + "="*95)
    print(f"  STARTING PATHVQA FOUNDATION EVALUATION: {model_key.upper()} (flaviagiammarino/path-vqa)")
    print("="*95)

    model, canonical_name = get_vqa_model(model_key)

    train_loader, val_loader, test_loader, seed, test_samples = create_path_vqa_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        seed=active_seed,
        subsample=args.subsample
    )

    print(f"\n[Evaluator] Running unseen holdout test evaluation on {len(test_loader.dataset)} QA pairs...")
    predictions = []
    ground_truths = []
    is_closed_flags = []
    first_vis_sample = None

    for batch in tqdm(test_loader, desc=f"Evaluating {canonical_name}"):
        pil_images = batch["pil_image"]
        questions = batch["question"]
        answers = batch["answer"]
        is_closed = batch["is_closed"]

        for im, q, a, c in zip(pil_images, questions, answers, is_closed):
            pred = model.answer_question(im, q)
            predictions.append(pred)
            ground_truths.append(a)
            is_closed_flags.append(c)

            if first_vis_sample is None:
                first_vis_sample = (im, q, a, pred)

    metrics = evaluate_vqa_predictions(predictions, ground_truths, is_closed_flags)
    metrics["Experiment_Seed"] = int(seed)
    metrics["Model_Name"] = canonical_name

    print_vqa_report(metrics, canonical_name, seed=seed)

    # Save summary JSON
    json_path = results_dir / f"metrics_summary_{canonical_name}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Main] Metrics saved to: {json_path}")

    # Save diagnostic visual QA card
    if first_vis_sample is not None:
        vis_img, vis_q, vis_gt, vis_pred = first_vis_sample
        diag_path = results_dir / f"vqa_diagnostic_{canonical_name}.png"
        plot_vqa_diagnostic_card(
            vis_img, vis_q, vis_gt, vis_pred,
            output_path=diag_path,
            model_name=canonical_name
        )

    return metrics


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    active_seed = setup_random_seed(args.seed)

    if args.model_name == "all":
        models = ["biomedclip", "paligemma", "qwen2vl", "blip2"]
        print(f"[Main] Running full PathVQA benchmark across all {len(models)} Foundation Models...")
        for m in models:
            run_vqa_pipeline(m, args, project_root, active_seed)
    else:
        run_vqa_pipeline(args.model_name, args, project_root, active_seed)


if __name__ == "__main__":
    main()

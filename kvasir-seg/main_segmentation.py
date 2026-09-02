"""
Main CLI Entrypoint for Foundation Semantic Segmentation on Hugging Face Kvasir-SEG.
Supports 4 Foundation Models:
1. medsam (MedSAM / Segment Anything Model)
2. segformer (NVIDIA SegFormer-B3)
3. convnext_unet (Meta ConvNeXt-Base U-Net)
4. swin_unet (Microsoft Swin-Base U-Net)
5. all (runs all 4 sequentially)
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim

from hf_kvasir_dataset import create_kvasir_dataloaders, setup_random_seed
from seg_metrics import CombinedBceDiceLoss, compute_segmentation_metrics, print_segmentation_report, plot_4panel_segmentation_grid
from model_medsam import MedSAMSegmentationModel
from model_segformer import SegFormerSegmentationModel
from model_convnext_unet import ConvNeXtUNetModel
from model_swin_unet import SwinUNetModel


def parse_args():
    parser = argparse.ArgumentParser(description="Kvasir-SEG Foundation Semantic Segmentation")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")
    parser.add_argument(
        "--model-name", type=str, default="segformer",
        choices=["medsam", "segformer", "convnext_unet", "swin_unet", "all"],
        help="Foundation segmentation model to train & evaluate"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Dynamic experiment seed")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample dataset for fast testing")
    return parser.parse_args()


def get_model(model_name: str):
    name_lower = model_name.lower()
    if "medsam" in name_lower or "sam" in name_lower:
        return MedSAMSegmentationModel(), "medsam"
    elif "segformer" in name_lower:
        return SegFormerSegmentationModel(), "segformer"
    elif "convnext" in name_lower:
        return ConvNeXtUNetModel(), "convnext_unet"
    elif "swin" in name_lower:
        return SwinUNetModel(), "swin_unet"
    else:
        print(f"[Warning] Unknown model '{model_name}', defaulting to SegFormer...")
        return SegFormerSegmentationModel(), "segformer"


def run_segmentation_pipeline(model_key: str, args, project_root: Path, active_seed: int):
    results_dir = Path(args.results_dir) if Path(args.results_dir).is_absolute() else project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else project_root / args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*95)
    print(f"  STARTING FOUNDATION SEGMENTATION PIPELINE: {model_key.upper()} (KVASIR-SEG)")
    print("="*95)

    model, canonical_name = get_model(model_key)
    model.to(device)

    train_loader, val_loader, test_loader, seed, test_samples = create_kvasir_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        seed=active_seed,
        subsample=args.subsample
    )

    criterion = CombinedBceDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = -1.0
    best_ckpt_path = results_dir / f"best_model_{canonical_name}.pt"

    print(f"\n[Trainer] Training {canonical_name} for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = masks.cpu().numpy()

                for p, t in zip(probs, targets):
                    m = compute_segmentation_metrics(p[0], t[0])
                    val_dices.append(m["Dice_Coefficient"])

        val_mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} - Val Mean Dice: {val_mean_dice*100:.2f}%")

        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            torch.save(model.state_dict(), best_ckpt_path)

    # Load best checkpoint for final 15% unseen holdout test evaluation
    if best_ckpt_path.exists():
        print(f"[Trainer] Loading best checkpoint from {best_ckpt_path} (Val Dice: {best_val_dice*100:.2f}%)...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    model.eval()
    print(f"\n[Evaluator] Running final holdout test evaluation on {len(test_loader.dataset)} samples...")
    test_metrics_list = []
    first_vis_sample = None

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {canonical_name}"):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            pil_imgs = batch["pil_image"]
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()

            for p, t, pil_im in zip(probs, targets, pil_imgs):
                m = compute_segmentation_metrics(p[0], t[0])
                test_metrics_list.append(m)
                if first_vis_sample is None:
                    first_vis_sample = (pil_im, t[0], p[0])

    # Average metrics
    avg_metrics = {
        "Dice_Coefficient": float(np.mean([m["Dice_Coefficient"] for m in test_metrics_list])),
        "mIoU_Jaccard": float(np.mean([m["mIoU_Jaccard"] for m in test_metrics_list])),
        "Precision": float(np.mean([m["Precision"] for m in test_metrics_list])),
        "Recall_Sensitivity": float(np.mean([m["Recall_Sensitivity"] for m in test_metrics_list])),
        "Specificity": float(np.mean([m["Specificity"] for m in test_metrics_list])),
        "Pixel_Accuracy": float(np.mean([m["Pixel_Accuracy"] for m in test_metrics_list])),
        "Experiment_Seed": int(seed),
        "Model_Name": canonical_name
    }

    print_segmentation_report(avg_metrics, canonical_name, seed=seed)

    # Save metrics summary JSON
    json_path = results_dir / f"metrics_summary_{canonical_name}.json"
    with open(json_path, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"[Main] Metrics saved to: {json_path}")

    # Save 4-panel diagnostic segmentation image
    if first_vis_sample is not None:
        vis_img, vis_gt, vis_pred = first_vis_sample
        overlay_path = results_dir / f"segmentation_analysis_{canonical_name}.png"
        plot_4panel_segmentation_grid(
            vis_img, vis_gt, vis_pred,
            output_path=overlay_path,
            title=f"{canonical_name.upper()} Polyp Segmentation Grid (DSC: {avg_metrics['Dice_Coefficient']*100:.2f}%)"
        )

    return avg_metrics


def main():
    args = parse_args()
    project_root = Path(__file__).parent.resolve()
    active_seed = setup_random_seed(args.seed)

    if args.model_name == "all":
        models = ["medsam", "segformer", "convnext_unet", "swin_unet"]
        print(f"[Main] Running full segmentation benchmark across all {len(models)} Foundation Models...")
        for m in models:
            run_segmentation_pipeline(m, args, project_root, active_seed)
    else:
        run_segmentation_pipeline(args.model_name, args, project_root, active_seed)


if __name__ == "__main__":
    main()

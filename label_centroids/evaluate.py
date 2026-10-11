import os
import argparse
import torch
import numpy as np
import pandas as pd

from data.dataset import get_dataloaders
from models.baseline_cnn import BaselineCNN
from models.lacnet import LACNet
from utils.metrics import (
    compute_metrics,
    compute_class_wise_metrics,
    format_table2_markdown
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LACNet or Baseline CNN")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--model_type", type=str, default="lacnet",
                        choices=["lacnet", "baseline_cnn", "without_refinement", "static_centroids", "without_attractor_reg"],
                        help="Type of model to evaluate")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to Kather colorectal dataset directory")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Image spatial resolution")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save class-wise performance as CSV")
    return parser.parse_args()


def evaluate_checkpoint(checkpoint_path: str, model_type: str = "lacnet", dataset_root: str = None,
                        batch_size: int = 64, image_size: int = 128, output_csv: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _, test_loader, _, test_ds = get_dataloaders(
        root_dir=dataset_root,
        batch_size=batch_size,
        image_size=image_size
    )

    if model_type in ["baseline_cnn", "softmax_classifier"]:
        model = BaselineCNN(num_classes=8).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    else:
        trainable = (model_type != "static_centroids")
        model = LACNet(num_classes=8, trainable_attractors=trainable).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                preds = model.predict(images)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    class_metrics = compute_class_wise_metrics(y_true, y_pred)

    print("\n================ Classification Metrics ================")
    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1']:.3f}")
    print(f"Macro Precision:  {metrics['macro_precision']:.3f}")
    print(f"Macro Recall:     {metrics['macro_recall']:.3f}")

    print("\n================ Table 2: Class-Wise Performance ================")
    table2_md = format_table2_markdown(class_metrics)
    print(table2_md)

    if output_csv:
        rows = []
        for r in class_metrics["classes"]:
            rows.append({
                "Class": r["class_name"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-score": r["f1_score"]
            })
        m = class_metrics["macro_avg"]
        rows.append({
            "Class": m["class_name"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-score": m["f1_score"]
        })
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved Table 2 CSV to: {output_csv}")

    return metrics, class_metrics


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_csv=args.output_csv
    )

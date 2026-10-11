import os
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from data.dataset import CLASS_DESCRIPTIONS, find_dataset_root, get_colorectal_data_splits
from models.lacnet import LACNet


def parse_args():
    parser = argparse.ArgumentParser(description="LACNet Inference and Zero-Shot Demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/lacnet_best.pth",
                        help="Path to trained LACNet checkpoint")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to single histology image (if None, selects random test sample)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to Kather colorectal dataset directory")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of random samples to evaluate if image_path is None")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first.")
    model = LACNet(num_classes=8).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LACNet model from: {args.checkpoint} (device: {device})...")
    model = load_model(args.checkpoint, device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    if args.image_path:
        samples = [(args.image_path, -1)]
    else:
        print("Selecting random test samples...")
        _, _, test_files, test_labels = get_colorectal_data_splits(root_dir=args.dataset_root)
        indices = random.sample(range(len(test_files)), min(args.num_samples, len(test_files)))
        samples = [(test_files[i], test_labels[i]) for i in indices]

    print("\n" + "="*70)
    print("           LACNet NEAREST-ATTRACTOR INFERENCE DEMO")
    print("="*70)

    for i, (path, true_label) in enumerate(samples, 1):
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode(tensor)  # (1, d)
            norm_attractors = F.normalize(model.label_attractors, p=2, dim=-1)  # (8, d)
            cos_sims = torch.matmul(emb, norm_attractors.t()).squeeze(0)  # (8,)
            pred_idx = torch.argmax(cos_sims).item()

        true_str = CLASS_DESCRIPTIONS[true_label] if true_label >= 0 else "Unknown"
        pred_str = CLASS_DESCRIPTIONS[pred_idx]
        status = "CORRECT" if true_label == pred_idx else "INCORRECT"

        print(f"\n--- Sample {i}: {os.path.basename(path)} ---")
        if true_label >= 0:
            print(f"  Ground Truth: Class {true_label} ({true_str})")
        print(f"  Predicted:    Class {pred_idx} ({pred_str}) [{status}]")
        print("  Cosine Similarities to Label Attractors:")
        for c in range(len(CLASS_DESCRIPTIONS)):
            sim = cos_sims[c].item()
            bar = "#" * int(max(0, (sim + 1) * 15))
            indicator = " <-- Pred" if c == pred_idx else ""
            print(f"    Class {c} ({CLASS_DESCRIPTIONS[c]:<25}): {sim:+.4f} | {bar:<30}{indicator}")

    print("\n" + "="*70)
    print("Demo completed.")


if __name__ == "__main__":
    args = parse_args()
    run_demo(args)

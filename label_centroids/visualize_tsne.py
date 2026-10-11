import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data.dataset import get_dataloaders
from models.baseline_cnn import BaselineCNN
from models.lacnet import LACNet
from utils.visualization import plot_tsne, CLASS_COLORS, CLASS_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate t-SNE visualizations (Figures 3 & 4)")
    parser.add_argument("--cnn_checkpoint", type=str, default="./checkpoints/baseline_cnn_best.pth",
                        help="Path to Baseline CNN checkpoint")
    parser.add_argument("--lacnet_checkpoint", type=str, default="./checkpoints/lacnet_best.pth",
                        help="Path to LACNet checkpoint")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to Kather colorectal dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save t-SNE figures")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for t-SNE reproducibility")
    return parser.parse_args()


def extract_embeddings(model, test_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            embs = model.get_embeddings(images)
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_labels


def generate_tsne_plots(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    _, test_loader, _, test_ds = get_dataloaders(root_dir=args.dataset_root)
    print(f"Loaded {len(test_ds)} test samples for t-SNE extraction.")

    # 1. Baseline CNN t-SNE (Figure 3)
    if os.path.exists(args.cnn_checkpoint):
        print("\nLoading Baseline CNN model...")
        cnn_model = BaselineCNN(num_classes=8).to(device)
        cnn_ckpt = torch.load(args.cnn_checkpoint, map_location=device, weights_only=False)
        cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])

        cnn_embs, labels = extract_embeddings(cnn_model, test_loader, device)
        cnn_fig_path = os.path.join(args.output_dir, "fig3_cnn_tsne.png")
        plot_tsne(
            cnn_embs,
            labels,
            title="CNN Embeddings t-SNE",
            save_path=cnn_fig_path,
            perplexity=args.perplexity,
            random_state=args.seed
        )
    else:
        print(f"Warning: Baseline CNN checkpoint not found at: {args.cnn_checkpoint}")

    # 2. LACNet t-SNE (Figure 4)
    if os.path.exists(args.lacnet_checkpoint):
        print("\nLoading LACNet model...")
        lacnet_model = LACNet(num_classes=8).to(device)
        lacnet_ckpt = torch.load(args.lacnet_checkpoint, map_location=device, weights_only=False)
        lacnet_model.load_state_dict(lacnet_ckpt["model_state_dict"])

        lacnet_embs, labels = extract_embeddings(lacnet_model, test_loader, device)
        lacnet_fig_path = os.path.join(args.output_dir, "fig4_lacnet_tsne.png")
        plot_tsne(
            lacnet_embs,
            labels,
            title="LACNet Embeddings t-SNE",
            save_path=lacnet_fig_path,
            perplexity=args.perplexity,
            random_state=args.seed
        )
    else:
        print(f"Warning: LACNet checkpoint not found at: {args.lacnet_checkpoint}")

    print("\nt-SNE generation completed successfully!")


if __name__ == "__main__":
    args = parse_args()
    generate_tsne_plots(args)

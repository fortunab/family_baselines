import os
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data.dataset import get_dataloaders
from models.baseline_cnn import BaselineCNN
from models.lacnet import LACNet
from utils.metrics import compute_metrics, compute_class_wise_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train LACNet or Baseline CNN on Colorectal Histology")
    parser.add_argument("--model_type", type=str, default="lacnet",
                        choices=["lacnet", "baseline_cnn", "without_refinement",
                                 "static_centroids", "without_attractor_reg", "softmax_classifier"],
                        help="Model variant to train")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Path to Kather colorectal dataset directory")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for mini-batch training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension d")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature tau for contrastive similarity")
    parser.add_argument("--lambda_reg", type=float, default=1e-4,
                        help="Attractor L2 regularization coefficient")
    parser.add_argument("--refine_interval", type=int, default=5,
                        help="Epoch interval for periodic centroid refinement")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Image spatial resolution")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_baseline(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, (correct / total) * 100.0


def evaluate_baseline(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["test_loss"] = total_loss / len(all_labels)
    return metrics, all_preds, all_labels


def train_epoch_lacnet(model, train_loader, optimizer, device, custom_lambda=None):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        embeddings, similarities = model(images)
        loss, ce_loss, reg_loss = model.compute_loss(similarities, labels, custom_lambda=custom_lambda)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_ce_loss += ce_loss.item() * images.size(0)
        total_reg_loss += reg_loss.item() * images.size(0)

        preds = torch.argmax(similarities, dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {
        "loss": total_loss / total,
        "ce_loss": total_ce_loss / total,
        "reg_loss": total_reg_loss / total,
        "accuracy": (correct / total) * 100.0
    }


def evaluate_lacnet(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model.predict(images)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_preds, all_labels


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_loader, test_loader, train_ds, test_ds = get_dataloaders(
        root_dir=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed
    )
    print(f"Data loaded: {len(train_ds)} train samples, {len(test_ds)} test samples.")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"{args.model_type}_best.pth")
    history_path = os.path.join(args.save_dir, f"{args.model_type}_history.json")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_macro_f1": []
    }

    best_test_acc = 0.0
    best_metrics = {}

    if args.model_type in ["baseline_cnn", "softmax_classifier"]:
        print("\n================ Training Baseline CNN ================")
        model = BaselineCNN(num_classes=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch_baseline(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            test_metrics, _, _ = evaluate_baseline(model, test_loader, criterion, device)
            elapsed = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_metrics["accuracy"])
            history["test_macro_f1"].append(test_metrics["macro_f1"])

            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_metrics['accuracy']:.2f}%, Test Macro F1: {test_metrics['macro_f1']:.3f}")

            if test_metrics["accuracy"] > best_test_acc:
                best_test_acc = test_metrics["accuracy"]
                best_metrics = test_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": test_metrics,
                    "args": vars(args)
                }, checkpoint_path)

    else:
        # LACNet and its ablation variants
        print(f"\n================ Training LACNet ({args.model_type}) ================")
        trainable_attractors = (args.model_type != "static_centroids")
        use_refinement = (args.model_type != "without_refinement")
        lambda_reg = 0.0 if args.model_type == "without_attractor_reg" else args.lambda_reg

        model = LACNet(
            num_classes=8,
            embedding_dim=args.embedding_dim,
            temperature=args.temperature,
            lambda_reg=lambda_reg,
            trainable_attractors=trainable_attractors
        ).to(device)

        resume_path = getattr(args, "resume", None)
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming LACNet from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_test_acc = ckpt.get("metrics", {}).get("accuracy", 0.0)
            best_metrics = ckpt.get("metrics", {})
            print(f"Resumed checkpoint baseline test acc: {best_test_acc:.2f}%")
        else:
            # Initialize attractors with class-wise centroids of initial embeddings (Section 3.3)
            print("Initializing label attractors from class-wise initial centroids...")
            model.initialize_attractors(train_loader, device)

        # AdamW optimizer as specified in Section 3.5
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_res = train_epoch_lacnet(model, train_loader, optimizer, device, custom_lambda=lambda_reg)
            scheduler.step()

            # Periodic Attractor Refinement (Section 3.5)
            if use_refinement and (epoch % args.refine_interval == 0):
                print(f"--> [Epoch {epoch}] Executing periodic attractor refinement...")
                model.periodic_refinement(train_loader, device)

            test_metrics, _, _ = evaluate_lacnet(model, test_loader, device)
            elapsed = time.time() - t0

            history["epoch"].append(epoch)
            history["train_loss"].append(train_res["loss"])
            history["train_acc"].append(train_res["accuracy"])
            history["test_acc"].append(test_metrics["accuracy"])
            history["test_macro_f1"].append(test_metrics["macro_f1"])

            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
                  f"Loss: {train_res['loss']:.4f} (CE: {train_res['ce_loss']:.4f}, Reg: {train_res['reg_loss']:.4f}) | "
                  f"Test Acc: {test_metrics['accuracy']:.2f}%, Test Macro F1: {test_metrics['macro_f1']:.3f}")

            if test_metrics["accuracy"] > best_test_acc:
                best_test_acc = test_metrics["accuracy"]
                best_metrics = test_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": test_metrics,
                    "args": vars(args)
                }, checkpoint_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best Test Accuracy: {best_test_acc:.2f}%, Macro F1: {best_metrics.get('macro_f1', 0.0):.3f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    return model, best_metrics


if __name__ == "__main__":
    args = parse_args()
    train_model(args)

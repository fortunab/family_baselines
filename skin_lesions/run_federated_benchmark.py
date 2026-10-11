"""Master CLI Runner for Federated Skin Lesion Classification (FedML / W&B / TOML / INI).

Executes:
1. Loads configuration from either TOML or INI (configparser)
2. Builds / verifies HAM10000 7-class dermoscopy dataset
3. Performs Non-IID Dirichlet partitioning across clinical hospital nodes
4. Initializes Weights & Biases (W&B) experiment tracking (online or offline)
5. Executes Federated Learning communication rounds (FedProx / FedAvg)
6. Computes comprehensive clinical dermatology metrics
7. Generates publication figures and exports JSON benchmark summaries
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from src.client import FederatedClient
from src.config_loader import load_config
from src.dataset_generator import build_ham10000_dermoscopy_dataset
from src.federated_partitioner import (
    partition_dataset_dirichlet,
    save_partition_manifest,
)
from src.server import FederatedServer
from src.visualizer import (
    plot_dermoscopy_7class_gallery,
    plot_dermoscopy_confusion_matrix,
    plot_federated_convergence_curves,
    plot_malignancy_screening_roc_frontier,
    plot_non_iid_dirichlet_distribution,
)
from src.wandb_logger import FederatedWandbLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FederatedBenchmark")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Federated Skin Lesion Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/federated_config.toml",
        help="Path to configuration file (.toml or .ini/.cfg)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["fedprox", "fedavg", "centralized", "local_isolated"],
        help="Federated algorithm override",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Communication rounds override")
    parser.add_argument("--mu", type=float, default=None, help="FedProx proximal mu weight override")
    parser.add_argument("--alpha", type=float, default=None, help="Non-IID Dirichlet concentration alpha override")
    parser.add_argument(
        "--wandb-mode", type=str, default=None, choices=["offline", "online", "disabled"], help="W&B tracking mode"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configuration (TOML or INI)
    logger.info(f"Loading configuration from {args.config}...")
    overrides = {}
    if args.algorithm:
        overrides["algorithm"] = args.algorithm
    if args.rounds:
        overrides["rounds"] = args.rounds
    if args.mu is not None:
        overrides["fedprox_mu"] = args.mu
    if args.alpha is not None:
        overrides["dirichlet_alpha"] = args.alpha
    if args.wandb_mode:
        overrides["mode"] = args.wandb_mode
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = load_config(args.config, overrides=overrides)

    out_root = Path(cfg.experiment.output_dir)
    fig_dir = out_root / "figures"
    metrics_dir = out_root / "metrics"
    models_dir = out_root / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build or Load Dataset
    logger.info("Initializing HAM10000 7-class dermoscopy dataset...")
    t_data_start = time.perf_counter()
    samples, manifest_meta = build_ham10000_dermoscopy_dataset(
        output_dir="data",
        total_samples=cfg.dataset.total_samples,
        image_size=cfg.dataset.image_size,
        test_split_ratio=cfg.dataset.test_split_ratio,
        seed=cfg.experiment.seed,
    )
    t_data = time.perf_counter() - t_data_start
    logger.info(f"Dataset generated in {t_data:.2f}s: {len(samples)} total scans.")

    test_samples = [s for s in samples if s.split == "test"]
    logger.info(f"Held-out global clinical test cohort: {len(test_samples)} scans.")

    # 3. Non-IID Dirichlet Partition across Hospitals
    logger.info(
        f"Partitioning data across {cfg.federated.num_clients} hospitals with Dirichlet alpha = {cfg.federated.dirichlet_alpha}..."
    )
    client_partitions, client_label_matrix = partition_dataset_dirichlet(
        samples=samples,
        num_clients=cfg.federated.num_clients,
        client_names=cfg.federated.client_names,
        alpha=cfg.federated.dirichlet_alpha,
        seed=cfg.experiment.seed,
    )
    save_partition_manifest(client_partitions, output_path="data/manifests/federated_partitions.json")

    # 4. Initialize W&B Logger
    wandb_logger = FederatedWandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config_dict=cfg.to_dict(),
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        enabled=cfg.wandb.enabled,
    )
    run_name = f"{cfg.experiment.algorithm.upper()}_alpha{cfg.federated.dirichlet_alpha}_r{cfg.federated.rounds}"
    wandb_logger.init_run(run_name=run_name)

    # 5. Initialize Server & Clients
    server = FederatedServer(
        test_samples=test_samples,
        model_arch=cfg.model.architecture,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        device=cfg.experiment.device,
        class_names=cfg.dataset.classes,
        malignant_classes=cfg.dataset.malignant_classes,
    )

    clients: List[FederatedClient] = []
    for cid, part in client_partitions.items():
        client_samples = [samples[idx] for idx in part.sample_indices]
        client_obj = FederatedClient(
            client_id=cid,
            client_name=part.client_name,
            samples=client_samples,
            model_arch=cfg.model.architecture,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            batch_size=cfg.federated.batch_size,
            local_epochs=cfg.federated.local_epochs,
            algorithm=cfg.experiment.algorithm,
            fedprox_mu=cfg.federated.fedprox_mu,
            device=cfg.experiment.device,
        )
        clients.append(client_obj)

    # 6. Execute Federated Communication Rounds
    logger.info("\n" + "=" * 90)
    logger.info(
        f"STARTING FEDERATED LEARNING: Algorithm={cfg.experiment.algorithm.upper()} | "
        f"Rounds={cfg.federated.rounds} | Clients={len(clients)}"
    )
    logger.info("=" * 90)

    t_fed_start = time.perf_counter()
    final_global_metrics = {}

    for round_idx in range(1, cfg.federated.rounds + 1):
        t_r_start = time.perf_counter()
        global_m, client_m = server.run_communication_round(round_idx, clients)
        t_r = time.perf_counter() - t_r_start

        # Log round to W&B
        wandb_logger.log_round_metrics(round_idx, global_m, client_m)

        final_global_metrics = global_m

        # Console Progress
        ref = global_m["binary_malignancy_referral"]
        print(
            f"Round {round_idx:02d}/{cfg.federated.rounds:02d} ({t_r:.2f}s) | "
            f"Loss: {global_m['test_loss']:.4f} | "
            f"Acc: {global_m['accuracy']*100:.2f}% | "
            f"BalAcc: {global_m['balanced_accuracy']*100:.2f}% | "
            f"Kappa: {global_m['cohen_kappa']:.4f} | "
            f"MacroF1: {global_m['macro_f1']*100:.2f}% | "
            f"MaligSens: {ref['referral_sensitivity']*100:.2f}% | "
            f"MaligSpec: {ref['referral_specificity']*100:.2f}%"
        )

    t_fed_total = time.perf_counter() - t_fed_start
    logger.info(f"Federated training completed in {t_fed_total:.2f}s.")

    # 7. Save Checkpoint
    checkpoint_path = models_dir / f"{cfg.experiment.name}_{cfg.experiment.algorithm}.pt"
    server.save_checkpoint(str(checkpoint_path))

    # 8. Generate Visualizations
    logger.info("Rendering publication diagnostic figures...")

    # Figure 1: Convergence Curves
    fig_conv = fig_dir / "federated_convergence_curves.png"
    plot_federated_convergence_curves(server.history, fig_conv, algorithm_name=cfg.experiment.algorithm.upper())
    logger.info(f"Saved convergence curves to {fig_conv}")

    # Figure 2: Dirichlet Label Skew
    fig_diri = fig_dir / "non_iid_dirichlet_distribution.png"
    plot_non_iid_dirichlet_distribution(
        client_label_matrix,
        cfg.federated.client_names,
        cfg.dataset.classes,
        fig_diri,
        alpha=cfg.federated.dirichlet_alpha,
    )
    logger.info(f"Saved Dirichlet distribution figure to {fig_diri}")

    # Figure 3: Confusion Matrix
    fig_cm = fig_dir / "dermoscopy_confusion_matrix.png"
    plot_dermoscopy_confusion_matrix(final_global_metrics["confusion_matrix"], cfg.dataset.classes, fig_cm)
    logger.info(f"Saved confusion matrix to {fig_cm}")

    # Figure 4: Malignancy Screening ROC
    # Re-evaluate server with probabilities for ROC
    _, detailed_metrics = server.evaluate_global_model()
    # Compute test probabilities on test set
    server.global_model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for imgs, tgts in server.test_loader:
            logits = server.global_model(imgs.to(server.device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_targets.append(tgts.numpy())
    test_probs = np.concatenate(all_probs, axis=0)
    test_tgts = np.concatenate(all_targets, axis=0)

    fig_roc = fig_dir / "malignancy_screening_roc.png"
    plot_malignancy_screening_roc_frontier(
        test_tgts,
        test_probs,
        cfg.dataset.classes,
        cfg.dataset.malignant_classes,
        fig_roc,
    )
    logger.info(f"Saved ROC figure to {fig_roc}")

    # Figure 5: 7-Class Dermoscopic Gallery
    fig_gal = fig_dir / "dermoscopy_7class_gallery.png"
    plot_dermoscopy_7class_gallery(samples, fig_gal)
    logger.info(f"Saved 7-class gallery to {fig_gal}")

    # 9. Log Figures to W&B
    wandb_logger.log_image_artifact(str(fig_conv), "Federated Convergence Trajectory")
    wandb_logger.log_image_artifact(str(fig_diri), "Non-IID Dirichlet Allocation")
    wandb_logger.log_image_artifact(str(fig_cm), "7-Class Confusion Matrix")
    wandb_logger.log_image_artifact(str(fig_roc), "Malignancy Screening ROC")
    wandb_logger.log_image_artifact(str(fig_gal), "7-Class Dermoscopic Gallery")

    # 10. Copy Figures to Brain Artifact Directory
    brain_artifact_dir = Path("C:/Users/Lenovo/.gemini/antigravity/brain/8b0e4da9-b9f6-4ac6-9d0c-010d087df687")
    if brain_artifact_dir.exists():
        for fig_f in [fig_conv, fig_diri, fig_cm, fig_roc, fig_gal]:
            dest = brain_artifact_dir / fig_f.name
            shutil.copy2(fig_f, dest)
            logger.info(f"Copied figure to artifact directory: {dest.name}")

    # 11. Export JSON Summary
    summary_export = {
        "experiment_name": cfg.experiment.name,
        "algorithm": cfg.experiment.algorithm,
        "communication_rounds": cfg.federated.rounds,
        "num_clients": cfg.federated.num_clients,
        "client_names": cfg.federated.client_names,
        "dirichlet_alpha": cfg.federated.dirichlet_alpha,
        "fedprox_mu": cfg.federated.fedprox_mu,
        "total_training_time_sec": round(t_fed_total, 2),
        "final_metrics": final_global_metrics,
        "best_cohen_kappa": round(server.best_kappa, 4),
        "history": server.history,
    }
    json_path = metrics_dir / "federated_benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_export, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {json_path}")

    # 12. Final W&B Summary & Close
    wandb_logger.log_summary(
        {
            "final_accuracy": final_global_metrics["accuracy"],
            "final_balanced_accuracy": final_global_metrics["balanced_accuracy"],
            "final_cohen_kappa": final_global_metrics["cohen_kappa"],
            "final_macro_f1": final_global_metrics["macro_f1"],
            "best_cohen_kappa": server.best_kappa,
            "total_training_time_sec": round(t_fed_total, 2),
        }
    )
    wandb_logger.finish()

    # 13. Formatted Console Leaderboard
    print("\n" + "=" * 105)
    print("        FEDERATED CLINICAL DERMATOLOGY BENCHMARK: FINAL GLOBAL CONSENSUS SUMMARY        ")
    print("=" * 105)
    print(f"Algorithm:                    {cfg.experiment.algorithm.upper()}")
    print(f"Clinical Hospitals:           {', '.join(cfg.federated.client_names)}")
    print(f"Non-IID Dirichlet Alpha:      {cfg.federated.dirichlet_alpha}")
    print(f"Communication Rounds:         {cfg.federated.rounds}")
    print(f"7-Class Accuracy:             {final_global_metrics['accuracy']*100:.2f}%")
    print(f"Balanced Accuracy:            {final_global_metrics['balanced_accuracy']*100:.2f}%")
    print(f"Cohen's Kappa (Agreement):    {final_global_metrics['cohen_kappa']:.4f}")
    print(f"Macro F1-Score:               {final_global_metrics['macro_f1']*100:.2f}%")
    ref_final = final_global_metrics["binary_malignancy_referral"]
    print(f"Malignancy Referral Sens:     {ref_final['referral_sensitivity']*100:.2f}%  (Target: >= 90%)")
    print(f"Malignancy Referral Spec:     {ref_final['referral_specificity']*100:.2f}%")
    print(f"Expected Calibration Error:   {final_global_metrics['expected_calibration_error']:.4f}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()

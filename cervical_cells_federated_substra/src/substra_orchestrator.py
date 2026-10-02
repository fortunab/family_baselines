"""
Substra Compute Plan DAG Orchestrator & Federated Aggregation Engine for Cervical Cytology.
Orchestrates multi-organization topology (Hospital Nodes + Coordinator Node),
supports dual-framework (fastai + skorch) local execution, FedAvg/FedProx/FedAdam aggregation,
and performs centralized holdout evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.evaluator import evaluate_multiclass_metrics
from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    get_model_parameters,
    set_model_parameters,
)
from src.substra_algo import SubstraCervicalAlgo
from src.wandb_tracker import WandbExperimentTracker


class SubstraComputePlanOrchestrator:
    """
    Substra Compute Plan DAG Orchestrator for Cervical Cytology.
    Coordinates distributed training tasks across Substra hospital nodes
    and central aggregation on the coordinator node.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        client_dfs: list[pd.DataFrame],
        test_df: pd.DataFrame,
        tracker: WandbExperimentTracker,
        active_seed: int,
    ):
        self.cfg = cfg
        self.client_dfs = client_dfs
        self.test_df = test_df
        self.tracker = tracker
        self.active_seed = active_seed
        self.num_clients = len(client_dfs)
        self.strategy_name = cfg.get("strategy", "FedAvg")
        self.num_rounds = int(cfg.get("num_rounds", 10))
        self.framework = str(cfg.get("framework", "fastai")).lower()

        # Build Substra Hospital Organization Nodes
        self.org_nodes: list[SubstraCervicalAlgo] = []
        for i in range(self.num_clients):
            org_id = f"Hospital_Node_{i}"
            algo = SubstraCervicalAlgo(org_id=org_id, cfg=cfg)
            self.org_nodes.append(algo)

        # Build Central Coordinator Model
        arch_name = cfg.get("architecture", "convnext_small")
        num_classes = int(cfg.get("num_classes", 7))
        pretrained = bool(cfg.get("pretrained", True))
        dropout = float(cfg.get("dropout_rate", 0.2))
        self.global_model = build_cervical_model(
            arch_name=arch_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.global_weights = get_model_parameters(self.global_model)

        # FedAdam state buffers
        self.adam_m: list[np.ndarray] | None = None
        self.adam_v: list[np.ndarray] | None = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.adam_eta = 0.01
        self.adam_tau = 1e-3

    def aggregate_fedavg(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """Weighted federated averaging across participating hospital nodes."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.global_weights

        num_layers = len(self.global_weights)
        new_weights = [np.zeros_like(w, dtype=np.float64) for w in self.global_weights]

        for client_idx, c_weights in enumerate(client_weights):
            weight_factor = sample_counts[client_idx] / total_samples
            for layer_idx in range(num_layers):
                new_weights[layer_idx] += c_weights[layer_idx].astype(np.float64) * weight_factor

        return [w.astype(np.float32) for w in new_weights]

    def aggregate_fedadam(
        self,
        client_weights: list[list[np.ndarray]],
        sample_counts: list[int],
    ) -> list[np.ndarray]:
        """Adaptive server-side optimization with first and second moment tracking."""
        total_samples = sum(sample_counts)
        if total_samples == 0:
            return self.global_weights

        avg_weights = self.aggregate_fedavg(client_weights, sample_counts)

        if self.adam_m is None:
            self.adam_m = [np.zeros_like(w) for w in self.global_weights]
            self.adam_v = [np.zeros_like(w) for w in self.global_weights]

        new_global_weights = []
        for l_idx in range(len(self.global_weights)):
            w = self.global_weights[l_idx]
            delta = avg_weights[l_idx] - w

            self.adam_m[l_idx] = self.adam_beta1 * self.adam_m[l_idx] + (1 - self.adam_beta1) * delta
            self.adam_v[l_idx] = self.adam_beta2 * self.adam_v[l_idx] + (1 - self.adam_beta2) * (delta**2)

            v_hat = np.sqrt(self.adam_v[l_idx]) + self.adam_tau
            step = self.adam_eta * (self.adam_m[l_idx] / v_hat)
            new_w = w + step
            new_global_weights.append(new_w.astype(np.float32))

        return new_global_weights

    def run_compute_plan(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Executes Substra Compute Plan DAG across rounds:
        For round r = 1..R:
          1. Broadcast current global weights to Substra hospital organization nodes
          2. Execute local training tasks in parallel across hospital nodes (fastai or skorch)
          3. Aggregate client updates at central coordinator (FedAvg / FedProx / FedAdam)
          4. Execute central evaluation task on holdout test partition
          5. Log metrics to W&B and console
        """
        print(
            f"\n[Substra-DAG] Launching Compute Plan: {self.num_rounds} Rounds across "
            f"{self.num_clients} Hospital Nodes ({self.strategy_name} | {self.framework.upper()})"
        )
        print(f"[Substra-DAG] Active Dynamic Seed: {self.active_seed} (No Seed 42)")

        # Initial Round 0 Evaluation
        set_model_parameters(self.global_model, self.global_weights)
        y_true_0, y_prob_0, loss_0 = evaluate_model_on_test_set(self.global_model, self.test_df, self.cfg)
        r0_metrics = evaluate_multiclass_metrics(y_true_0, y_prob_0, loss_0)
        self.tracker.log_round(0, r0_metrics)
        print(
            f"[Substra Round 0/Holdout] Accuracy: {r0_metrics['accuracy'] * 100:.2f}% | "
            f"Macro F1: {r0_metrics['macro_f1']:.4f} | Loss: {loss_0:.4f}"
        )

        for round_idx in range(1, self.num_rounds + 1):
            print(f"\n--- [Substra Compute Plan Round {round_idx}/{self.num_rounds}] ---")
            client_weights = []
            sample_counts = []
            round_losses = []

            # Step 1 & 2: Local Substra node training tasks
            for client_id, algo in enumerate(self.org_nodes):
                c_df = self.client_dfs[client_id]
                w_up, loss_val, n_samples = algo.train(
                    data=c_df,
                    in_model_weights=self.global_weights,
                )
                client_weights.append(w_up)
                sample_counts.append(n_samples)
                round_losses.append(loss_val)

            avg_local_loss = float(np.mean(round_losses)) if round_losses else 0.0

            # Step 3: Central aggregation task
            if self.strategy_name == "FedAdam":
                self.global_weights = self.aggregate_fedadam(client_weights, sample_counts)
            else:
                # FedAvg and FedProx both use weighted model averaging on server
                self.global_weights = self.aggregate_fedavg(client_weights, sample_counts)

            set_model_parameters(self.global_model, self.global_weights)

            # Step 4: Centralized evaluation task on holdout test set
            y_true, y_prob, test_loss = evaluate_model_on_test_set(
                self.global_model,
                self.test_df,
                self.cfg,
            )
            round_metrics = evaluate_multiclass_metrics(y_true, y_prob, test_loss)
            round_metrics["avg_local_train_loss"] = avg_local_loss

            # Step 5: Logging
            self.tracker.log_round(round_idx, round_metrics)
            print(
                f"[Substra Holdout Eval R{round_idx}] Accuracy: {round_metrics['accuracy'] * 100:.2f}% | "
                f"BalAcc: {round_metrics['balanced_accuracy'] * 100:.2f}% | "
                f"Macro F1: {round_metrics['macro_f1']:.4f} | "
                f"ROC-AUC: {round_metrics['macro_roc_auc']:.4f} | Test Loss: {test_loss:.4f}"
            )

        # Final predictions on holdout set
        y_true_final, y_prob_final, final_loss = evaluate_model_on_test_set(
            self.global_model,
            self.test_df,
            self.cfg,
        )
        final_metrics = evaluate_multiclass_metrics(y_true_final, y_prob_final, final_loss)
        return y_true_final, y_prob_final, final_metrics

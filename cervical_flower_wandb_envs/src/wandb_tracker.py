"""
Weights & Biases (W&B) and Local CSV/JSON Experiment Telemetry Tracker for Cervical Flower FL.
Provides offline fallback, round-by-round metric tracking, and artifact logging.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class WandbExperimentTracker:
    """
    Cervical Flower Federated Experiment Tracker interfacing with Weights & Biases.
    Automatically handles offline modes and creates local persistent backups.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("wandb_enabled", cfg.get("enabled", True)))
        self.project = cfg.get("project", "cervical-cells-federated-flower")
        self.entity = cfg.get("entity", None) or None
        self.run_name = cfg.get("run_name", "cervical_flower_run")
        self.mode = cfg.get("mode", "online")
        self.output_dir = Path(cfg.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.round_history: List[Dict[str, Any]] = []
        self.run = None

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb

            if not os.environ.get("WANDB_API_KEY") and self.mode == "online":
                print("[W&B-Tracker] No WANDB_API_KEY detected. Falling back to offline mode.")
                self.mode = "offline"

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.cfg,
                mode=self.mode,
                reinit=True,
            )
            print(f"[W&B-Tracker] Initialized run '{self.run_name}' in mode: {self.mode}")
        except Exception as e:
            print(f"[W&B-Tracker] Warning: Failed to initialize Weights & Biases: {e}")
            print("[W&B-Tracker] Local CSV/JSON tracking will remain active.")
            self.enabled = False

    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Logs metrics evaluated at a Flower federated communication round."""
        record = {"round": round_num}
        record.update(metrics)
        self.round_history.append(record)

        # 1. Weights & Biases logging
        if self.enabled and self.run is not None:
            try:
                import wandb

                wandb_payload = {f"server/{k}": v for k, v in metrics.items()}
                wandb_payload["round"] = round_num
                wandb.log(wandb_payload, step=round_num)
            except Exception as e:
                print(f"[W&B-Tracker] Warning: Logging to W&B failed: {e}")

        # 2. Local CSV persistent backup
        csv_path = self.output_dir / "federated_training_history.csv"
        df = pd.DataFrame(self.round_history)
        df.to_csv(csv_path, index=False)

    def log_artifact_file(
        self, file_path: str, artifact_name: str, artifact_type: str = "plot"
    ) -> None:
        """Uploads a visual artifact (ROC curve, Confusion matrix) to W&B."""
        if not self.enabled or self.run is None:
            return
        try:
            import wandb

            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            print(f"[W&B-Tracker] Logged artifact '{artifact_name}' to W&B.")
        except Exception as e:
            print(f"[W&B-Tracker] Failed to upload artifact '{artifact_name}': {e}")

    def save_final_summary(self, summary_metrics: Dict[str, Any]) -> None:
        """Saves experiment configuration and final evaluation metrics to JSON."""
        exclude_keys = {"y_true", "y_pred", "y_prob"}
        clean_metrics = {
            k: v
            for k, v in summary_metrics.items()
            if k not in exclude_keys and isinstance(v, (int, float, str, bool))
        }

        summary_payload = {
            "config": {k: str(v) for k, v in self.cfg.items()},
            "final_metrics": clean_metrics,
            "round_history": self.round_history,
        }
        json_path = self.output_dir / "federated_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"[W&B-Tracker] Saved final summary to: {json_path}")

    def finish(self) -> None:
        """Closes W&B run cleanly."""
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
                print(f"[W&B-Tracker] Tracking session '{self.run_name}' finished.")
            except Exception:
                pass

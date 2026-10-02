"""
Weights & Biases (W&B) MLOps Experiment Tracking for Substra Cervical Cytology.
Supports live cloud logging, offline mode, scalar serialization sanitization, and artifact exports.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def sanitize_wandb_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively converts NumPy numbers and arrays into standard Python types for W&B."""
    clean = {}
    for k, v in data.items():
        if isinstance(v, (np.floating, float)):
            clean[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, dict):
            clean[k] = sanitize_wandb_payload(v)
        else:
            clean[k] = v
    return clean


class WandbExperimentTracker:
    """
    Substra W&B Telemetry & Metrics Tracker.
    Safely captures round metrics, validation loss, and logs artifacts.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("wandb_enabled", True)) and WANDB_AVAILABLE
        self.project = cfg.get("project", "cervical-cells-federated-substra")
        self.run_name = cfg.get("run_name", "substra_cervical_fl")
        self.history: list[dict[str, Any]] = []
        self.run = None

        if self.enabled:
            mode = cfg.get("mode", "online")
            try:
                self.run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    config=sanitize_wandb_payload(cfg),
                    mode=mode,
                    reinit=True,
                )
                print(f"[W&B-Tracker] Initialized Weights & Biases run: {self.run_name} ({mode})")
            except Exception as e:
                print(f"[W&B-Tracker] Failed to connect to W&B ({e}). Switching to local offline tracking.")
                self.enabled = False

    def log_round(self, round_idx: int, metrics: dict[str, Any]) -> None:
        """Logs aggregated round metrics."""
        clean_metrics = sanitize_wandb_payload(metrics)
        payload = {"round": round_idx, **clean_metrics}
        self.history.append(payload)

        if self.enabled and self.run:
            try:
                wandb.log(payload)
            except Exception:
                pass

    def log_artifact(self, file_path: Path, artifact_type: str = "plot") -> None:
        """Uploads evaluation plot or summary artifact to W&B."""
        if self.enabled and self.run and file_path.exists():
            try:
                artifact = wandb.Artifact(name=file_path.stem, type=artifact_type)
                artifact.add_file(str(file_path))
                self.run.log_artifact(artifact)
            except Exception:
                pass

    def save_summary(self, output_dir: Path, final_metrics: dict[str, Any]) -> None:
        """Saves local JSON summary and CSV history."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "federated_summary.json"
        history_path = output_dir / "federated_training_history.csv"

        clean_final = sanitize_wandb_payload(final_metrics)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(clean_final, f, indent=2)

        if self.history:
            df_hist = pd.DataFrame(self.history)
            df_hist.to_csv(history_path, index=False)

        print(f"[W&B-Tracker] Saved final summary to: {summary_path}")

    def finish(self) -> None:
        """Closes W&B run cleanly."""
        if self.enabled and self.run:
            try:
                wandb.finish()
            except Exception:
                pass

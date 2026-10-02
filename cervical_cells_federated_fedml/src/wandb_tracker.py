"""
Weights & Biases (W&B) MLOps Experiment Tracker for FedML Cytology.
Supports online tracking, offline fallback, and local JSON/CSV metrics export.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sanitize_metric_value(val: Any) -> Any:
    """Converts NumPy or PyTorch scalar types into JSON-serializable Python primitives."""
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val)


class WandbExperimentTracker:
    """
    MLOps Experiment Tracker managing W&B metrics telemetry,
    artifact logging, and local metric persistence.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("wandb_enabled", False))
        self.project = cfg.get("project", "cervical-cells-federated-fedml")
        self.run_name = cfg.get("run_name", f"fedml_{cfg.get('framework', 'fastai')}_{cfg.get('strategy', 'FedAvg')}")
        self.mode = cfg.get("mode", "online")
        self.run = None
        self.history: list[dict[str, Any]] = []

        if self.enabled:
            try:
                import wandb

                self.run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    mode=self.mode,
                    config=cfg,
                    reinit=True,
                )
                print(f"[W&B] Run initialized: {self.run_name} (project: {self.project})")
            except Exception as e:
                print(f"[W&B] Warning: Failed to initialize W&B ({e}). Continuing in offline mode.")
                self.enabled = False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Logs scalar metrics to W&B and accumulates in internal history."""
        sanitized = {k: sanitize_metric_value(v) for k, v in metrics.items()}
        if step is not None:
            sanitized["step"] = step

        self.history.append(sanitized)

        if self.enabled and self.run is not None:
            try:
                import wandb

                wandb.log(sanitized, step=step)
            except Exception as e:
                print(f"[W&B] Warning logging metrics: {e}")

    def log_artifact(self, file_path: str | Path, name: str, artifact_type: str = "evaluation-plot") -> None:
        """Uploads a local file artifact to W&B."""
        p = Path(file_path)
        if not p.exists():
            return

        if self.enabled and self.run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(name=name, type=artifact_type)
                artifact.add_file(str(p.resolve()))
                self.run.log_artifact(artifact)
            except Exception as e:
                print(f"[W&B] Warning uploading artifact {name}: {e}")

    def export_summary(self, output_dir: str | Path, filename_prefix: str = "fedml") -> None:
        """Exports accumulated metrics history to JSON and CSV formats."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_path = out_dir / f"{filename_prefix}_metrics_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # Export CSV
        if self.history:
            csv_path = out_dir / f"{filename_prefix}_metrics_history.csv"
            df = pd.DataFrame(self.history)
            df.to_csv(csv_path, index=False)
            print(f"[Metrics] Saved metrics history to {json_path} and {csv_path}")

    def finish(self) -> None:
        """Closes W&B run session."""
        if self.enabled and self.run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception as e:
                print(f"[W&B] Warning finishing run: {e}")

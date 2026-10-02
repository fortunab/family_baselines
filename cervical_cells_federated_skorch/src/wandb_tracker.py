"""
Weights & Biases (W&B) Experiment Tracking and Offline Telemetry Module.
Guarantees clean scalar logging, JSON serialization, and graceful offline fallback.
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


class WandbExperimentTracker:
    """Manages experiment metrics, live dashboard updates, and artifact persistence."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("wandb_enabled", cfg.get("enabled", True)))
        self.project = cfg.get("project", "cervical-cells-federated-skorch")
        self.run_name = cfg.get("run_name", "cervical_skorch_fedavg")
        self.mode = cfg.get("mode", "online")
        self.run = None
        self.history: list[dict[str, Any]] = []

        if self.enabled:
            try:
                import wandb

                os.environ["WANDB_SILENT"] = "true"
                self.run = wandb.init(
                    project=self.project,
                    name=self.run_name,
                    mode=self.mode,
                    config=self.cfg,
                    reinit=True,
                )
                print(f"[W&B-Tracker] Initialized run '{self.run_name}' in project '{self.project}'.")
            except Exception as e:
                print(f"[W&B-Tracker] Note: Live W&B logging unavailable ({e}). Continuing in local mode.")
                self.enabled = False

    def log_round_metrics(
        self,
        round_num: int,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Logs global server-side round metrics to W&B and internal history."""
        clean_metrics = {"round": round_num}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean_metrics[k] = float(v)

        self.history.append(clean_metrics)

        if self.enabled and self.run is not None:
            try:
                import wandb

                wandb.log(clean_metrics, step=step if step is not None else round_num)
            except Exception:
                pass

    def log_artifact(
        self,
        file_path: Path | str,
        artifact_type: str = "visual-metric",
    ) -> None:
        """Persists visualization plots or checkpoint files into W&B artifacts."""
        path = Path(file_path)
        if not path.exists():
            return

        if self.enabled and self.run is not None:
            try:
                import wandb

                artifact = wandb.Artifact(
                    name=f"{self.run_name}_{path.stem}",
                    type=artifact_type,
                )
                artifact.add_file(str(path))
                self.run.log_artifact(artifact)
            except Exception:
                pass

    def save_final_summary(
        self,
        output_dir: Path | str,
        final_metrics: dict[str, Any],
    ) -> None:
        """Saves a JSON summary and CSV history to the local results directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        serializable_metrics = {}
        for k, v in final_metrics.items():
            if isinstance(v, (int, float, str, bool)):
                serializable_metrics[k] = v
            elif hasattr(v, "item"):
                try:
                    serializable_metrics[k] = v.item()
                except Exception:
                    pass

        summary_file = out / "federated_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2)

        if self.history:
            df_hist = pd.DataFrame(self.history)
            df_hist.to_csv(out / "federated_training_history.csv", index=False)

        print(f"[W&B-Tracker] Saved final summary to: {summary_file}")

    def finish(self) -> None:
        """Flushes telemetry buffers and finalizes the W&B run."""
        if self.enabled and self.run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

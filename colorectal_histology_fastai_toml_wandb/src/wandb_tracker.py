"""
Deep Weights & Biases (W&B) Experiment Tracking Integration for fastai.
Reference: https://wandb.ai/broutonlab/first_steps/reports/Data-Science-Experiments-Management-with-Weights-Biases---Vmlldzo2NjE3MDI
"""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class WandbExperimentTracker:
    def __init__(
        self,
        project_name: str = "colorectal-histology-fastai",
        experiment_name: str = "fastai_run",
        config_dict: Optional[Dict[str, Any]] = None,
        results_dir: Optional[Path] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config_dict = config_dict or {}
        self.results_dir = Path(results_dir or "./results").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tags = tags or ["fastai", "histology", "toml"]
        self.enabled = enabled

        self.wandb_run = None
        self.local_history: List[Dict[str, Any]] = []

        self.csv_path = self.results_dir / f"telemetry_{self.experiment_name}.csv"
        self.json_path = self.results_dir / f"telemetry_{self.experiment_name}.json"

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self):
        print(f"[W&B-Tracker] Initializing Weights & Biases for project: '{self.project_name}'...")
        try:
            import wandb

            # If no API key is found, set offline mode to avoid runtime authentication prompts
            if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
                os.environ["WANDB_MODE"] = "offline"
                print(
                    "[W&B-Tracker] Notice: No WANDB_API_KEY detected. Running in W&B 'offline' mode."
                )

            # Filter raw unpicklable keys
            clean_cfg = {
                k: v
                for k, v in self.config_dict.items()
                if not str(k).startswith("_") and isinstance(v, (int, float, str, bool, list))
            }

            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=clean_cfg,
                tags=self.tags,
                reinit=True,
            )
            print(f"[W&B-Tracker] W&B Run active: {wandb.run.name} (ID: {wandb.run.id})")
        except Exception as e:
            print(
                f"[W&B-Tracker] W&B initialization notice ({e}). Continuing with local telemetry logging."
            )
            self.wandb_run = None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        step_entry = {"step": step, **metrics}
        self.local_history.append(step_entry)

        if self.wandb_run is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)
            except Exception:
                pass

        # Always save local telemetry CSV & JSON
        try:
            file_exists = self.csv_path.exists()
            keys = list(step_entry.keys())
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(step_entry)

            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self.local_history, f, indent=4)
        except Exception:
            pass

    def log_artifact(self, file_path: Path, artifact_type: str = "evaluation_result"):
        file_path = Path(file_path)
        if not file_path.exists():
            return

        if self.wandb_run is not None:
            try:
                import wandb

                art = wandb.Artifact(name=f"artifact-{self.experiment_name}", type=artifact_type)
                art.add_file(str(file_path))
                wandb.log_artifact(art)
                print(f"[W&B-Tracker] Uploaded artifact '{file_path.name}' to W&B.")
            except Exception as e:
                print(f"[W&B-Tracker] Artifact upload notice: {e}")

    def finish(self):
        if self.wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        print(f"[W&B-Tracker] Tracking session '{self.experiment_name}' finished.")

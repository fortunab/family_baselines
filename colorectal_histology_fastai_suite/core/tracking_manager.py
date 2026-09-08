"""
Unified Experiment Tracking & MLOps Engine.
Supports:
1. Weights & Biases (W&B) - https://wandb.ai/
2. MLflow - https://mlflow.org/
3. TensorBoard - torch.utils.tensorboard
4. Local / Offline Fallback JSON & CSV Telemetry Logger
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentTracker:
    def __init__(
        self,
        backend: str = "offline",
        project: str = "colorectal-histology",
        experiment_name: str = "run",
        config_dict: Optional[Dict[str, Any]] = None,
        results_dir: Optional[Path] = None,
        tags: Optional[str] = None
    ):
        self.backend = (backend or "offline").lower()
        self.project = project
        self.experiment_name = experiment_name
        self.config_dict = config_dict or {}
        self.results_dir = Path(results_dir or "./results").resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tags = [t.strip() for t in tags.split(",")] if tags else []

        self.wandb_run = None
        self.mlflow_run = None
        self.tb_writer = None
        self.local_history = []

        self._init_backend()

    def _init_backend(self):
        print(f"[Tracker] Initializing experiment tracker with backend: '{self.backend}'...")

        # 1. Weights & Biases (W&B)
        if self.backend in ("wandb", "weights_and_biases", "w&b"):
            try:
                import wandb
                # If no API key is set, enable offline mode to prevent crash
                if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
                    os.environ["WANDB_MODE"] = "offline"
                    print("[Tracker] Notice: No WANDB_API_KEY found, running W&B in 'offline' mode.")

                self.wandb_run = wandb.init(
                    project=self.project,
                    name=self.experiment_name,
                    config=self.config_dict,
                    tags=self.tags,
                    reinit=True
                )
                print(f"[Tracker] W&B run initialized: {wandb.run.name} (ID: {wandb.run.id})")
            except Exception as e:
                print(f"[Tracker] W&B initialization notice ({e}), enabling offline local fallback.")
                self.backend = "offline"

        # 2. MLflow
        elif self.backend == "mlflow":
            try:
                import mlflow
                mlflow.set_experiment(self.project)
                self.mlflow_run = mlflow.start_run(run_name=self.experiment_name)
                # Log hyperparameters
                clean_params = {k: v for k, v in self.config_dict.items() if not str(k).startswith("_") and isinstance(v, (int, float, str, bool))}
                mlflow.log_params(clean_params)
                print(f"[Tracker] MLflow run initialized: {self.mlflow_run.info.run_id}")
            except Exception as e:
                print(f"[Tracker] MLflow initialization notice ({e}), enabling offline local fallback.")
                self.backend = "offline"

        # 3. TensorBoard
        elif self.backend in ("tensorboard", "tb"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.results_dir / "tensorboard_logs" / self.experiment_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                print(f"[Tracker] TensorBoard SummaryWriter initialized at: {tb_dir}")
            except Exception as e:
                print(f"[Tracker] TensorBoard initialization notice ({e}), enabling offline local fallback.")
                self.backend = "offline"

        # 4. Local / Offline Logger (Always active as ground-truth telemetry)
        self.csv_log_path = self.results_dir / f"telemetry_{self.experiment_name}.csv"
        self.json_log_path = self.results_dir / f"telemetry_{self.experiment_name}.json"
        print(f"[Tracker] Local telemetry logs will be persisted to: {self.csv_log_path}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        step_entry = {"step": step, **metrics}
        self.local_history.append(step_entry)

        # 1. Log to W&B
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception:
                pass

        # 2. Log to MLflow
        if self.mlflow_run is not None:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v), step=step)
            except Exception:
                pass

        # 3. Log to TensorBoard
        if self.tb_writer is not None:
            try:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(k, float(v), global_step=step)
            except Exception:
                pass

        # 4. Log to local CSV & JSON
        try:
            self._append_to_csv(step_entry)
            with open(self.json_log_path, "w") as f:
                json.dump(self.local_history, f, indent=4)
        except Exception:
            pass

    def _append_to_csv(self, step_entry: Dict[str, Any]):
        file_exists = self.csv_log_path.exists()
        keys = list(step_entry.keys())
        with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow(step_entry)

    def log_artifact(self, artifact_path: Path):
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            return

        if self.wandb_run is not None:
            try:
                import wandb
                art = wandb.Artifact(name=f"artifact-{self.experiment_name}", type="model_evaluation")
                art.add_file(str(artifact_path))
                wandb.log_artifact(art)
            except Exception:
                pass

        if self.mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_artifact(str(artifact_path))
            except Exception:
                pass

    def finish(self):
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        if self.mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass

        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass

        print(f"[Tracker] Experiment '{self.experiment_name}' tracking finished. Local telemetry saved.")

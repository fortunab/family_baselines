"""Weights & Biases (W&B) Logging Manager for Federated Diabetic Retinopathy.

Supports both online cloud logging and self-contained offline mode (WANDB_MODE=offline).
Logs communication rounds, QWK metrics, client drift, and diagnostic figure artifacts.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("DRWandbLogger")

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class FederatedWandbLogger:
    """Manages experiment tracking with Weights & Biases for federated DR."""

    def __init__(
        self,
        project: str = "dr-fedml-ophthalmology",
        entity: Optional[str] = None,
        mode: str = "offline",
        config_dict: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.project = project
        self.entity = entity if entity else None
        self.mode = mode
        self.config_dict = config_dict or {}
        self.tags = tags or ["federated-learning", "diabetic-retinopathy", "eyepacs"]
        self.notes = notes
        self.run = None

        if not WANDB_AVAILABLE and enabled:
            logger.warning("wandb package not installed. Experiment will run without W&B tracking.")

    def init_run(self, run_name: Optional[str] = None):
        """Initializes the W&B run."""
        if not self.enabled:
            logger.info("W&B tracking is disabled or wandb is not installed.")
            return

        if self.mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_SILENT"] = "true"
        else:
            os.environ["WANDB_MODE"] = "online"

        try:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=self.config_dict,
                tags=self.tags,
                notes=self.notes,
                mode=self.mode,
                reinit=True,
            )
            logger.info(f"W&B Run initialized: name='{self.run.name}', id='{self.run.id}', mode='{self.mode}'")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B run: {e}. Continuing without W&B.")
            self.enabled = False

    def log_round_metrics(
        self,
        round_idx: int,
        global_metrics: Dict[str, Any],
        client_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Logs metrics for a single communication round."""
        if not self.enabled or self.run is None:
            return

        payload: Dict[str, Any] = {
            "round": round_idx,
            "global/accuracy": global_metrics.get("accuracy", 0.0),
            "global/balanced_accuracy": global_metrics.get("balanced_accuracy", 0.0),
            "global/quadratic_weighted_kappa": global_metrics.get("quadratic_weighted_kappa", 0.0),
            "global/macro_f1": global_metrics.get("macro_f1", 0.0),
            "global/referral_sensitivity": global_metrics.get("binary_referral", {}).get("referral_sensitivity", 0.0),
            "global/referral_specificity": global_metrics.get("binary_referral", {}).get("referral_specificity", 0.0),
            "global/ece": global_metrics.get("expected_calibration_error", 0.0),
        }

        if client_metrics:
            for c_id, c_data in client_metrics.items():
                payload[f"client_{c_id}/train_loss"] = c_data.get("train_loss", 0.0)
                payload[f"client_{c_id}/divergence"] = c_data.get("divergence", 0.0)

        try:
            wandb.log(payload, step=round_idx)
        except Exception as e:
            logger.warning(f"Error logging to W&B: {e}")

    def log_summary(self, summary_dict: Dict[str, Any]):
        """Logs summary values to W&B."""
        if not self.enabled or self.run is None:
            return

        for k, v in summary_dict.items():
            if isinstance(v, (int, float, str, bool)):
                self.run.summary[k] = v

    def log_image_artifact(self, image_path: str, caption: str = "Diagnostic Figure"):
        """Logs an image artifact to W&B."""
        if not self.enabled or self.run is None:
            return
        p = Path(image_path)
        if p.exists():
            try:
                wandb.log({p.stem: wandb.Image(str(p), caption=caption)})
            except Exception as e:
                logger.warning(f"Failed to log image artifact to W&B: {e}")

    def finish(self):
        """Finalizes the W&B run."""
        if self.enabled and self.run is not None:
            try:
                wandb.finish()
                logger.info("W&B Run successfully completed and closed.")
            except Exception as e:
                logger.warning(f"Error closing W&B run: {e}")

"""
Substra Federated Algorithm (Algo / Function) for Cervical Cytology.
Supports dual-framework execution: fastai (1-Cycle policy) and skorch (Scikit-Learn API).
Executes privacy-preserving local training within Substra organization nodes.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fastai_engine import (
    build_cervical_model,
    evaluate_model_on_test_set,
    local_train_fastai,
    set_model_parameters,
)
from src.skorch_engine import local_train_skorch
from src.substra_opener import SubstraCervicalOpener


class SubstraCervicalAlgo:
    """
    Substra Algorithm asset executing training and evaluation inside an organization node.
    Enforces privacy by performing compute locally without transferring patient data.
    Supports both fastai and skorch engines.
    """

    def __init__(self, org_id: str, cfg: dict[str, Any]):
        self.org_id = org_id
        self.cfg = cfg
        self.framework = str(cfg.get("framework", "fastai")).lower()
        self.opener = SubstraCervicalOpener(image_size=int(cfg.get("image_size", 224)))

        # Build local model instance (uninitialized; global coordinator broadcasts weights)
        arch_name = cfg.get("architecture", "convnext_small")
        num_classes = int(cfg.get("num_classes", 7))
        dropout = float(cfg.get("dropout_rate", 0.2))
        self.model = build_cervical_model(
            arch_name=arch_name,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )

    def train(
        self,
        data: pd.DataFrame | str | Path,
        in_model_weights: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], float, int]:
        """
        Executes local Substra train task:
        1. Ingests global model parameters (if provided)
        2. Loads local clinical data via the Opener
        3. Trains local model with fastai (1-Cycle) or skorch (.fit)
        4. Returns updated model weights, training loss, and sample count.
        """
        client_df = self.opener.get_data(data)

        if in_model_weights is not None:
            set_model_parameters(self.model, in_model_weights)

        if self.framework == "skorch":
            updated_weights, loss, num_samples = local_train_skorch(
                model=self.model,
                client_df=client_df,
                cfg=self.cfg,
                global_model_weights=in_model_weights,
            )
        else:
            updated_weights, loss, num_samples = local_train_fastai(
                model=self.model,
                client_df=client_df,
                cfg=self.cfg,
                global_model_weights=in_model_weights,
            )

        return updated_weights, loss, num_samples

    def predict(
        self,
        data: pd.DataFrame | str | Path,
        in_model_weights: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Executes local Substra evaluation/predict task on local data.
        Returns: (y_true, y_prob, loss)
        """
        eval_df = self.opener.get_data(data)

        if in_model_weights is not None:
            set_model_parameters(self.model, in_model_weights)

        return evaluate_model_on_test_set(self.model, eval_df, self.cfg)

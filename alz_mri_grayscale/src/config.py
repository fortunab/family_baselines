"""Configuration Loader for Alzheimer MRI Grayscale Benchmark Suite."""

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": "Alzheimer_s_Dataset",
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed_grayscale",
        "image_size": 128,
        "foundation_image_size": 224,
        "test_size": 0.20,
        "val_size": 0.15,
        "seed": 42,
        "classes": [
            "NonDemented",
            "VeryMildDemented",
            "MildDemented",
            "ModerateDemented",
        ],
    },
    "preprocessing": {
        "convert_grayscale": True,
        "denoise_filter": "gaussian",
        "gaussian_sigma": 0.8,
        "contrast_enhancement": "clahe",
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": 8,
        "skull_strip_threshold": 0.05,
        "normalize_range": [0.0, 1.0],
    },
    "ml_family": {
        "models": [
            "svm_rbf",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "mlp",
            "logistic_regression",
            "voting_ensemble",
        ],
        "feature_modes": ["radiomic_texture_pca", "foundation_embeddings"],
        "pca_variance": 0.95,
        "n_estimators": 200,
    },
    "foundation_models": {
        "backbones": [
            "densenet121",
            "resnet50",
            "convnext_small",
            "efficientnet_b0",
        ],
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 0.001,
        "device": "cpu",
    },
    "output": {
        "results_dir": "results",
        "metrics_dir": "results/metrics",
        "export_plots": True,
    },
}


def get_default_config() -> Dict[str, Any]:
    """Returns a deep copy of the default configuration dictionary."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(config_path: Optional[Union[Path, str]] = None) -> Dict[str, Any]:
    """Loads YAML configuration file, falling back to default config."""
    cfg = get_default_config()
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if isinstance(v, dict) and k in cfg:
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
    return cfg


def merge_cli_args(cfg: Dict[str, Any], args: Optional[argparse.Namespace] = None, **kwargs) -> Dict[str, Any]:
    """Merges command line arguments into config dictionary."""
    merged = copy.deepcopy(cfg)
    cli_dict: Dict[str, Any] = {}
    if args is not None:
        cli_dict.update(vars(args))
    cli_dict.update(kwargs)

    if cli_dict.get("raw_data_dir"):
        merged["dataset"]["raw_data_dir"] = cli_dict["raw_data_dir"]
    if cli_dict.get("processed_data_dir"):
        merged["dataset"]["processed_data_dir"] = cli_dict["processed_data_dir"]
    if cli_dict.get("image_size") is not None:
        merged["dataset"]["image_size"] = cli_dict["image_size"]
    if cli_dict.get("seed") is not None:
        merged["dataset"]["seed"] = cli_dict["seed"]
    if cli_dict.get("device") is not None:
        merged["foundation_models"]["device"] = cli_dict["device"]

    return merged

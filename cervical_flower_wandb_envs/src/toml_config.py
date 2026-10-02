"""
TOML Configuration Loader and CLI Override Engine for Cervical Cytology Federated Learning.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import toml


def load_toml_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration dictionary from a TOML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = toml.load(f)
    return cfg


def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens nested TOML dictionary sections into top-level keys."""
    flat = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat[k] = v
        else:
            flat[section] = values
    return flat


def merge_cli_args(cfg: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merges CLI argument overrides into the loaded TOML configuration."""
    flat = flatten_config(cfg)

    # CLI Overrides
    if getattr(cli_args, "arch", None):
        flat["architecture"] = cli_args.arch
    if getattr(cli_args, "strategy", None):
        flat["strategy"] = cli_args.strategy
    if getattr(cli_args, "num_rounds", None) is not None:
        flat["num_rounds"] = int(cli_args.num_rounds)
    if getattr(cli_args, "num_clients", None) is not None:
        flat["num_clients"] = int(cli_args.num_clients)
    if getattr(cli_args, "local_epochs", None) is not None:
        flat["local_epochs"] = int(cli_args.local_epochs)
    if getattr(cli_args, "lr", None) is not None:
        flat["local_lr"] = float(cli_args.lr)
    if getattr(cli_args, "proximal_mu", None) is not None:
        flat["proximal_mu"] = float(cli_args.proximal_mu)
    if getattr(cli_args, "non_iid", False):
        flat["non_iid"] = True
    if getattr(cli_args, "dirichlet_alpha", None) is not None:
        flat["dirichlet_alpha"] = float(cli_args.dirichlet_alpha)
    if getattr(cli_args, "seed", None) is not None:
        flat["seed"] = int(cli_args.seed)
    if getattr(cli_args, "no_wandb", False):
        flat["enabled"] = False
        flat["wandb_enabled"] = False
    else:
        flat["wandb_enabled"] = flat.get("enabled", True)
    if getattr(cli_args, "subsample", None) is not None:
        flat["subsample"] = int(cli_args.subsample)

    return flat


def print_config_summary(cfg: Dict[str, Any]) -> None:
    """Prints a clean tabular summary of the active cervical federated experiment settings."""
    print("=" * 80)
    print("    CERVICAL CYTOLOGY FEDERATED FLOWER EXPERIMENT CONFIGURATION")
    print("=" * 80)
    sections = {
        "General": ["seed", "device", "output_dir"],
        "Dataset": [
            "name",
            "num_classes",
            "batch_size",
            "non_iid",
            "dirichlet_alpha",
            "test_split",
        ],
        "Model": ["architecture", "pretrained", "dropout_rate"],
        "Flower FL": [
            "strategy",
            "num_rounds",
            "num_clients",
            "local_epochs",
            "local_lr",
            "proximal_mu",
        ],
        "W&B MLOps": ["wandb_enabled", "project", "run_name", "mode"],
    }
    for sec_name, keys in sections.items():
        print(f"[{sec_name}]")
        for k in keys:
            if k in cfg:
                print(f"  {k:22s} : {cfg[k]}")
    print("=" * 80 + "\n")

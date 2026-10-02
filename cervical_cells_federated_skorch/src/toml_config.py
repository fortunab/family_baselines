"""
TOML Configuration Parsing and CLI Argument Merging Module.
Supports profile loading from configs/*.toml with command-line overrides.
"""

import argparse
from pathlib import Path
from typing import Any

import toml


def load_toml_config(config_path: str) -> dict[str, Any]:
    """Loads a TOML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return toml.load(f)


def flatten_config(nested_cfg: dict[str, Any]) -> dict[str, Any]:
    """Flattens nested TOML dictionary into top-level key-value mapping."""
    flat: dict[str, Any] = {}
    for section, val in nested_cfg.items():
        if isinstance(val, dict):
            for k, v in val.items():
                flat[k] = v
        else:
            flat[section] = val
    return flat


def merge_cli_args(
    cfg: dict[str, Any],
    cli_args: argparse.Namespace,
) -> dict[str, Any]:
    """Merges command-line overrides into configuration dictionary."""
    flat = flatten_config(cfg)
    cli_dict = vars(cli_args)

    for k, v in cli_dict.items():
        if v is not None and k != "config":
            flat[k] = v

    if cli_args.no_wandb:
        flat["wandb_enabled"] = False
        flat["enabled"] = False

    return flat


def print_config_summary(cfg: dict[str, Any]) -> None:
    """Prints a structured summary banner of the active experiment configuration."""
    print("=" * 80)
    print("       CERVICAL CYTOLOGY FEDERATED SKORCH CONFIGURATION")
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

    for section_name, keys in sections.items():
        print(f"[{section_name}]")
        for k in keys:
            if k in cfg:
                print(f"  {k:<22} : {cfg[k]}")
    print("=" * 80 + "\n")

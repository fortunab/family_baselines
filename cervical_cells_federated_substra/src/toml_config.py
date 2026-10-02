"""
TOML Configuration Parser & Structured Banner for Substra Cervical Cytology.
Supports layered TOML configs, CLI overrides, and dynamic parameter validation.
"""

from pathlib import Path
from typing import Any

import toml


def load_toml_config(config_path: Path) -> dict[str, Any]:
    """Loads and parses TOML configuration file into a flattened dictionary."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = toml.load(f)

    flat_cfg: dict[str, Any] = {}
    for section_name, section_values in raw_cfg.items():
        if isinstance(section_values, dict):
            for k, v in section_values.items():
                flat_cfg[k] = v
        else:
            flat_cfg[section_name] = section_values

    return flat_cfg


def merge_cli_arguments(
    cfg: dict[str, Any],
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    """Merges command-line arguments into configuration dictionary."""
    merged = dict(cfg)
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v
    return merged


def print_substra_configuration_banner(cfg: dict[str, Any], active_seed: int | None = None) -> None:
    """Prints structured terminal banner for Substra experiment reproducibility."""
    seed_str = (
        f"{active_seed} (Dynamic Random Best Select - No Seed 42)"
        if active_seed
        else str(cfg.get("seed", "random_best"))
    )
    framework_str = str(cfg.get("framework", "fastai")).upper()

    print("=" * 80)
    print("       SUBSTRA FEDERATED CERVICAL CYTOLOGY EXPERIMENT CONFIGURATION")
    print(f"       Dual Framework: fastai + skorch | Active Engine: {framework_str}")
    print("=" * 80)
    print("[General]")
    print(f"  seed                   : {seed_str}")
    print(f"  device                 : {cfg.get('device', 'cuda')}")
    print(f"  output_dir             : {cfg.get('output_dir', 'results')}")
    print(f"  framework              : {cfg.get('framework', 'fastai')}")
    print("[Dataset]")
    print(f"  name                   : {cfg.get('name', 'herlev_cervical_cytology')}")
    print(f"  num_classes            : {cfg.get('num_classes', 7)}")
    print(f"  batch_size             : {cfg.get('batch_size', 32)}")
    print(f"  non_iid                : {cfg.get('non_iid', False)}")
    print(f"  dirichlet_alpha        : {cfg.get('dirichlet_alpha', 0.5)}")
    print(f"  test_split             : {cfg.get('test_split', 0.15)}")
    print("[Model]")
    print(f"  architecture           : {cfg.get('architecture', 'convnext_small')}")
    print(f"  pretrained             : {cfg.get('pretrained', True)}")
    print(f"  dropout_rate           : {cfg.get('dropout_rate', 0.2)}")
    print("[Substra FL]")
    print(f"  strategy               : {cfg.get('strategy', 'FedAvg')}")
    print(f"  num_rounds             : {cfg.get('num_rounds', 10)}")
    print(f"  num_clients            : {cfg.get('num_clients', 5)}")
    print(f"  local_epochs           : {cfg.get('local_epochs', 2)}")
    print(f"  local_lr               : {cfg.get('local_lr', 0.0003)}")
    print(f"  proximal_mu            : {cfg.get('proximal_mu', 1.0)}")
    print("[W&B MLOps]")
    print(f"  wandb_enabled          : {cfg.get('wandb_enabled', True)}")
    print(f"  project                : {cfg.get('project', 'cervical-cells-federated-substra')}")
    print(f"  run_name               : {cfg.get('run_name', 'substra_cervical_fl')}")
    print(f"  mode                   : {cfg.get('mode', 'online')}")
    print("=" * 80 + "\n")

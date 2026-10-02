"""
TOML Configuration Loader & Experiment Banner Generator for FedML Cytology.
Supports hierarchical TOML configs and runtime CLI overrides.
"""

from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore


def load_toml_config(config_path: str | Path) -> dict[str, Any]:
    """Loads and parses a TOML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "rb") as f:
        raw_cfg = tomllib.load(f)

    # Flatten sections into flat dictionary
    flattened: dict[str, Any] = {}
    for section_name, section_dict in raw_cfg.items():
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                flattened[k] = v
        else:
            flattened[section_name] = section_dict

    return flattened


def merge_cli_args_into_config(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    """Merges command-line arguments into configuration dictionary."""
    if hasattr(args, "framework") and args.framework is not None:
        cfg["framework"] = args.framework
    if hasattr(args, "strategy") and args.strategy is not None:
        cfg["strategy"] = args.strategy
    if hasattr(args, "rounds") and args.rounds is not None:
        cfg["num_rounds"] = args.rounds
    if hasattr(args, "clients") and args.clients is not None:
        cfg["num_clients"] = args.clients
    if hasattr(args, "epochs") and args.epochs is not None:
        cfg["local_epochs"] = args.epochs
    if hasattr(args, "lr") and args.lr is not None:
        cfg["local_lr"] = args.lr
    if hasattr(args, "batch_size") and args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if hasattr(args, "device") and args.device is not None:
        cfg["device"] = args.device
    if hasattr(args, "no_wandb") and args.no_wandb:
        cfg["wandb_enabled"] = False
    if hasattr(args, "subsample") and args.subsample is not None:
        cfg["subsample_limit"] = args.subsample
    if hasattr(args, "seed") and args.seed is not None:
        cfg["seed"] = args.seed

    return cfg


def print_experiment_banner(cfg: dict[str, Any], title: str = "FEDML FEDERATED CYTOLOGY EXPERIMENT") -> None:
    """Prints a structured ASCII configuration banner."""
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    sections = {
        "General Setup": ["seed", "device", "output_dir", "framework"],
        "Dataset Details": ["name", "num_classes", "batch_size", "non_iid", "dirichlet_alpha", "test_split"],
        "Model Backbone": ["architecture", "pretrained", "dropout_rate"],
        "FedML Federated": ["strategy", "num_rounds", "num_clients", "local_epochs", "local_lr", "proximal_mu"],
        "MLOps & Tracking": ["wandb_enabled", "project", "run_name", "mode"],
    }

    for sec_name, keys in sections.items():
        print(f"[{sec_name}]")
        for k in keys:
            if k in cfg:
                print(f"  {k:<22} : {cfg[k]}")
    print("=" * 80 + "\n")

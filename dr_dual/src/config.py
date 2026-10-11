"""Configuration loader and settings manager."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_default_config_path() -> Path:
    """Returns absolute path to configs/default.yaml."""
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "configs" / "default.yaml"


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads configuration YAML file."""
    if config_path is None:
        p = get_default_config_path()
    else:
        p = Path(config_path)

    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

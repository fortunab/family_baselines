"""
Pure TOML Configuration Engine & Schema Validator for Herlev Foundation skorch Models.
Parses TOML profiles and applies dynamic CLI overrides.
"""

from pathlib import Path
from typing import Any, Dict, Optional


def load_toml_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"TOML configuration file not found at: {file_path}")

    try:
        import tomllib  # Built-in in Python 3.11+

        with open(file_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml

            with open(file_path, "r", encoding="utf-8") as f:
                return toml.load(f)
        except ImportError:
            raise ImportError(
                "Please install 'toml' via `pip install toml` to parse TOML configuration files."
            ) from None


def load_and_validate_config(
    config_path_str: str, cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    config_path = Path(config_path_str).resolve()
    print(f"[TOML-Config] Loading configuration profile: {config_path}")

    raw_cfg = load_toml_file(config_path)

    # Flatten nested dictionary for convenient parameter access
    flat_cfg: Dict[str, Any] = {}
    for section_name, section_dict in raw_cfg.items():
        if isinstance(section_dict, dict):
            for k, v in section_dict.items():
                flat_cfg[f"{section_name.lower()}.{k.lower()}"] = v
                flat_cfg[k.lower()] = v
        else:
            flat_cfg[section_name.lower()] = section_dict

    flat_cfg["_raw_toml"] = raw_cfg

    # Apply dynamic CLI overrides if provided
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                k_clean = k.lower().replace("-", "_")
                flat_cfg[k_clean] = v
                print(f"[TOML-Config] CLI Override -> {k_clean} = {v}")

    return flat_cfg


def print_toml_summary(cfg: Dict[str, Any]):
    print("\n" + "=" * 85)
    print("      ACTIVE HERLEV FOUNDATION SKORCH + TOML EXPERIMENT CONFIGURATION")
    print("=" * 85)
    raw = cfg.get("_raw_toml", {})
    if raw:
        for section, params in raw.items():
            print(f"[{section}]")
            if isinstance(params, dict):
                for k, v in params.items():
                    print(f"  • {k:<24} = {v}")
            else:
                print(f"  • {section:<24} = {params}")
    else:
        for k, v in cfg.items():
            if not k.startswith("_"):
                print(f"  • {k:<28} = {v}")
    print("=" * 85 + "\n")

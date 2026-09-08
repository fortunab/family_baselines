"""
Modular Configuration Loader Engine using Python's Standard `configparser.ConfigParser`.
Supports INI (.ini, .cfg), TOML (.toml), JSON (.json), and dynamic CLI overrides.
Reference: https://docs.python.org/3/library/configparser.html
"""

import os
import json
import configparser
from pathlib import Path
from typing import Dict, Any, Optional


def load_ini_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    parser.read(str(config_path), encoding="utf-8")
    
    cfg_dict = {}
    for section in parser.sections():
        cfg_dict[section] = {}
        for key, val in parser.items(section):
            # Parse booleans
            val_clean = val.strip().lower()
            if val_clean in ("true", "yes", "on"):
                cfg_dict[section][key] = True
            elif val_clean in ("false", "no", "off"):
                cfg_dict[section][key] = False
            else:
                # Try parsing numeric (int or float)
                try:
                    if "." in val or "e" in val.lower():
                        cfg_dict[section][key] = float(val)
                    else:
                        cfg_dict[section][key] = int(val)
                except ValueError:
                    cfg_dict[section][key] = val
    return cfg_dict


def load_toml_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        import tomllib  # Python 3.11+
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import toml
            with open(config_path, "r", encoding="utf-8") as f:
                return toml.load(f)
        except ImportError:
            raise ImportError("Neither 'tomllib' nor 'toml' is installed to parse .toml config.")


def load_json_config(config_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path_str: str, cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config_path = Path(config_path_str).resolve()
    suffix = config_path.suffix.lower()

    print(f"[ConfigLoader] Loading configuration from: {config_path} (Format: {suffix})")

    if suffix in (".ini", ".cfg", ".conf"):
        cfg = load_ini_config(config_path)
    elif suffix == ".toml":
        cfg = load_toml_config(config_path)
    elif suffix == ".json":
        cfg = load_json_config(config_path)
    else:
        print(f"[ConfigLoader] Unknown suffix '{suffix}', attempting configparser parsing...")
        cfg = load_ini_config(config_path)

    # Flatten helper dictionary for fast access
    flat_cfg: Dict[str, Any] = {}
    for sec, items in cfg.items():
        for k, v in items.items():
            flat_cfg[f"{sec.lower()}.{k.lower()}"] = v
            flat_cfg[k.lower()] = v

    # Store raw nested dictionary under 'sections'
    flat_cfg["_raw_sections"] = cfg

    # Apply CLI overrides if provided
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                k_clean = k.lower().replace("-", "_")
                flat_cfg[k_clean] = v
                print(f"[ConfigLoader] CLI Override -> {k_clean} = {v}")

    return flat_cfg


def print_config_summary(cfg: Dict[str, Any]):
    print("\n" + "="*85)
    print("                     ACTIVE EXPERIMENT CONFIGURATION")
    print("="*85)
    raw = cfg.get("_raw_sections", {})
    if raw:
        for section, params in raw.items():
            print(f"[{section}]")
            for k, v in params.items():
                print(f"  • {k:<24} = {v}")
    else:
        for k, v in cfg.items():
            if not k.startswith("_"):
                print(f"  • {k:<28} = {v}")
    print("="*85 + "\n")

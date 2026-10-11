"""Unified Configuration Loader for Federated Diabetic Retinopathy Suite.

Supports both TOML format (via Python 3.11+ tomllib or tomli fallback)
and INI / CFG format (via Python standard library configparser).
Provides a typed dataclass schema with strict validation.
"""

import configparser
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Robust TOML import (Python 3.11+ tomllib, or tomli fallback)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class ExperimentConfig:
    name: str = "diabetic_retinopathy_federated_benchmark"
    algorithm: str = "fedprox"
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "results"
    log_level: str = "INFO"


@dataclass
class DatasetConfig:
    name: str = "dr_dual_db_eyepacs_messidor"
    num_classes: int = 5
    total_samples: int = 400
    image_size: int = 64
    test_split_ratio: float = 0.20
    classes: List[str] = field(
        default_factory=lambda: [
            "No_DR",
            "Mild_NPDR",
            "Moderate_NPDR",
            "Severe_NPDR",
            "Proliferative_DR",
        ]
    )
    referral_threshold_grade: int = 2


@dataclass
class FederatedParamConfig:
    num_clients: int = 4
    client_names: List[str] = field(
        default_factory=lambda: [
            "TeleScreening_EyePACS_California",
            "Hospital_Topcon_Paris_Messidor",
            "RetinaInstitute_IDRiD_India",
            "PrimaryCare_APTOS_Asia",
        ]
    )
    rounds: int = 15
    local_epochs: int = 2
    batch_size: int = 16
    dirichlet_alpha: float = 0.5
    client_fraction: float = 1.0
    fedprox_mu: float = 0.01


@dataclass
class ModelConfig:
    architecture: str = "resnet_retina"
    in_channels: int = 3
    hidden_dim: int = 64
    dropout: float = 0.25
    num_classes: int = 5


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    lr: float = 0.002
    weight_decay: float = 0.0001
    momentum: float = 0.9
    lr_decay: float = 0.96


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "dr-fedml-ophthalmology"
    entity: str = ""
    mode: str = "offline"
    tags: List[str] = field(
        default_factory=lambda: [
            "federated-learning",
            "diabetic-retinopathy",
            "eyepacs",
            "messidor",
            "fedprox",
            "ophthalmology",
        ]
    )
    notes: str = "Federated Multi-Hospital Non-IID Fundus Benchmark on 5-Class Diabetic Retinopathy"


@dataclass
class ClinicalConfig:
    referral_threshold_grade: int = 2
    min_target_qwk: float = 0.80
    min_referral_sens: float = 0.95
    save_checkpoints: bool = True


@dataclass
class FederatedConfig:
    """Master configuration encompassing all sub-configurations."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    federated: FederatedParamConfig = field(default_factory=FederatedParamConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_list(val: Union[str, List[str]]) -> List[str]:
    """Helper to parse list from string or keep existing list."""
    if isinstance(val, list):
        return [str(item).strip() for item in val]
    if isinstance(val, str):
        return [item.strip() for item in val.split(",") if item.strip()]
    return []


def load_toml_config(file_path: Union[str, Path]) -> FederatedConfig:
    """Loads configuration from a TOML file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TOML configuration file not found: {path}")

    if tomllib is None:
        raise ImportError("tomllib (Python 3.11+) or tomli package is required to load TOML files.")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    cfg = FederatedConfig()

    if "experiment" in data:
        exp = data["experiment"]
        cfg.experiment = ExperimentConfig(
            name=exp.get("name", cfg.experiment.name),
            algorithm=exp.get("algorithm", cfg.experiment.algorithm),
            seed=int(exp.get("seed", cfg.experiment.seed)),
            device=exp.get("device", cfg.experiment.device),
            output_dir=exp.get("output_dir", cfg.experiment.output_dir),
            log_level=exp.get("log_level", cfg.experiment.log_level),
        )

    if "dataset" in data:
        ds = data["dataset"]
        cfg.dataset = DatasetConfig(
            name=ds.get("name", cfg.dataset.name),
            num_classes=int(ds.get("num_classes", cfg.dataset.num_classes)),
            total_samples=int(ds.get("total_samples", cfg.dataset.total_samples)),
            image_size=int(ds.get("image_size", cfg.dataset.image_size)),
            test_split_ratio=float(ds.get("test_split_ratio", cfg.dataset.test_split_ratio)),
            classes=_parse_list(ds.get("classes", cfg.dataset.classes)),
            referral_threshold_grade=int(ds.get("referral_threshold_grade", cfg.dataset.referral_threshold_grade)),
        )

    if "federated" in data:
        fed = data["federated"]
        cfg.federated = FederatedParamConfig(
            num_clients=int(fed.get("num_clients", cfg.federated.num_clients)),
            client_names=_parse_list(fed.get("client_names", cfg.federated.client_names)),
            rounds=int(fed.get("rounds", cfg.federated.rounds)),
            local_epochs=int(fed.get("local_epochs", cfg.federated.local_epochs)),
            batch_size=int(fed.get("batch_size", cfg.federated.batch_size)),
            dirichlet_alpha=float(fed.get("dirichlet_alpha", cfg.federated.dirichlet_alpha)),
            client_fraction=float(fed.get("client_fraction", cfg.federated.client_fraction)),
            fedprox_mu=float(fed.get("fedprox_mu", cfg.federated.fedprox_mu)),
        )

    if "model" in data:
        m = data["model"]
        cfg.model = ModelConfig(
            architecture=m.get("architecture", cfg.model.architecture),
            in_channels=int(m.get("in_channels", cfg.model.in_channels)),
            hidden_dim=int(m.get("hidden_dim", cfg.model.hidden_dim)),
            dropout=float(m.get("dropout", cfg.model.dropout)),
            num_classes=int(m.get("num_classes", cfg.model.num_classes)),
        )

    if "optimizer" in data:
        opt = data["optimizer"]
        cfg.optimizer = OptimizerConfig(
            type=opt.get("type", cfg.optimizer.type),
            lr=float(opt.get("lr", cfg.optimizer.lr)),
            weight_decay=float(opt.get("weight_decay", cfg.optimizer.weight_decay)),
            momentum=float(opt.get("momentum", cfg.optimizer.momentum)),
            lr_decay=float(opt.get("lr_decay", cfg.optimizer.lr_decay)),
        )

    if "wandb" in data:
        wb = data["wandb"]
        cfg.wandb = WandbConfig(
            enabled=bool(wb.get("enabled", cfg.wandb.enabled)),
            project=wb.get("project", cfg.wandb.project),
            entity=wb.get("entity", cfg.wandb.entity),
            mode=wb.get("mode", cfg.wandb.mode),
            tags=_parse_list(wb.get("tags", cfg.wandb.tags)),
            notes=wb.get("notes", cfg.wandb.notes),
        )

    if "clinical" in data:
        cl = data["clinical"]
        cfg.clinical = ClinicalConfig(
            referral_threshold_grade=int(cl.get("referral_threshold_grade", cfg.clinical.referral_threshold_grade)),
            min_target_qwk=float(cl.get("min_target_qwk", cfg.clinical.min_target_qwk)),
            min_referral_sens=float(cl.get("min_referral_sens", cfg.clinical.min_referral_sens)),
            save_checkpoints=bool(cl.get("save_checkpoints", cfg.clinical.save_checkpoints)),
        )

    return cfg


def load_ini_config(file_path: Union[str, Path]) -> FederatedConfig:
    """Loads configuration using Python standard library configparser."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"INI configuration file not found: {path}")

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")

    cfg = FederatedConfig()

    if "experiment" in parser:
        exp = parser["experiment"]
        cfg.experiment = ExperimentConfig(
            name=exp.get("name", cfg.experiment.name),
            algorithm=exp.get("algorithm", cfg.experiment.algorithm),
            seed=exp.getint("seed", cfg.experiment.seed),
            device=exp.get("device", cfg.experiment.device),
            output_dir=exp.get("output_dir", cfg.experiment.output_dir),
            log_level=exp.get("log_level", cfg.experiment.log_level),
        )

    if "dataset" in parser:
        ds = parser["dataset"]
        cfg.dataset = DatasetConfig(
            name=ds.get("name", cfg.dataset.name),
            num_classes=ds.getint("num_classes", cfg.dataset.num_classes),
            total_samples=ds.getint("total_samples", cfg.dataset.total_samples),
            image_size=ds.getint("image_size", cfg.dataset.image_size),
            test_split_ratio=ds.getfloat("test_split_ratio", cfg.dataset.test_split_ratio),
            classes=_parse_list(ds.get("classes", ",".join(cfg.dataset.classes))),
            referral_threshold_grade=ds.getint("referral_threshold_grade", cfg.dataset.referral_threshold_grade),
        )

    if "federated" in parser:
        fed = parser["federated"]
        cfg.federated = FederatedParamConfig(
            num_clients=fed.getint("num_clients", cfg.federated.num_clients),
            client_names=_parse_list(fed.get("client_names", ",".join(cfg.federated.client_names))),
            rounds=fed.getint("rounds", cfg.federated.rounds),
            local_epochs=fed.getint("local_epochs", cfg.federated.local_epochs),
            batch_size=fed.getint("batch_size", cfg.federated.batch_size),
            dirichlet_alpha=fed.getfloat("dirichlet_alpha", cfg.federated.dirichlet_alpha),
            client_fraction=fed.getfloat("client_fraction", cfg.federated.client_fraction),
            fedprox_mu=fed.getfloat("fedprox_mu", cfg.federated.fedprox_mu),
        )

    if "model" in parser:
        m = parser["model"]
        cfg.model = ModelConfig(
            architecture=m.get("architecture", cfg.model.architecture),
            in_channels=m.getint("in_channels", cfg.model.in_channels),
            hidden_dim=m.getint("hidden_dim", cfg.model.hidden_dim),
            dropout=m.getfloat("dropout", cfg.model.dropout),
            num_classes=m.getint("num_classes", cfg.model.num_classes),
        )

    if "optimizer" in parser:
        opt = parser["optimizer"]
        cfg.optimizer = OptimizerConfig(
            type=opt.get("type", cfg.optimizer.type),
            lr=opt.getfloat("lr", cfg.optimizer.lr),
            weight_decay=opt.getfloat("weight_decay", cfg.optimizer.weight_decay),
            momentum=opt.getfloat("momentum", cfg.optimizer.momentum),
            lr_decay=opt.getfloat("lr_decay", cfg.optimizer.lr_decay),
        )

    if "wandb" in parser:
        wb = parser["wandb"]
        cfg.wandb = WandbConfig(
            enabled=wb.getboolean("enabled", cfg.wandb.enabled),
            project=wb.get("project", cfg.wandb.project),
            entity=wb.get("entity", cfg.wandb.entity),
            mode=wb.get("mode", cfg.wandb.mode),
            tags=_parse_list(wb.get("tags", ",".join(cfg.wandb.tags))),
            notes=wb.get("notes", cfg.wandb.notes),
        )

    if "clinical" in parser:
        cl = parser["clinical"]
        cfg.clinical = ClinicalConfig(
            referral_threshold_grade=cl.getint("referral_threshold_grade", cfg.clinical.referral_threshold_grade),
            min_target_qwk=cl.getfloat("min_target_qwk", cfg.clinical.min_target_qwk),
            min_referral_sens=cl.getfloat("min_referral_sens", cfg.clinical.min_referral_sens),
            save_checkpoints=cl.getboolean("save_checkpoints", cfg.clinical.save_checkpoints),
        )

    return cfg


def load_config(file_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> FederatedConfig:
    """Unified entry point: loads from .toml or .ini/.cfg based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".toml"]:
        cfg = load_toml_config(path)
    elif suffix in [".ini", ".cfg"]:
        cfg = load_ini_config(path)
    else:
        raise ValueError(f"Unsupported configuration file extension '{suffix}'. Use .toml or .ini/.cfg.")

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg.experiment, k):
                setattr(cfg.experiment, k, v)
            elif hasattr(cfg.federated, k):
                setattr(cfg.federated, k, v)
            elif hasattr(cfg.wandb, k):
                setattr(cfg.wandb, k, v)

    return cfg

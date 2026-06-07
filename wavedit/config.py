"""Typed experiment configuration loaded from a single YAML file.

A whole training run is described by one readable YAML file (see ``configs/``).
The launcher scripts parse it into the nested :class:`Config` dataclass below, so
the rest of the codebase works with attribute access (``cfg.train.lr``) instead of
a wall of argparse flags. Configs round-trip to plain dicts so they can be embedded
in checkpoints, making every checkpoint self-contained for generation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .utils import get_logger

logger = get_logger(__name__)

FLOW_FORMULATIONS = ("rectified", "cfm", "ot_fm")
PRECISIONS = ("fp32", "fp16", "bf16")
SAMPLERS = ("heun", "euler")


@dataclass
class DataConfig:
    """Where the data lives and how conditions are interpreted."""

    metadata_csv: str | None = None        # CSV mode (recommended); takes priority
    data_folder: str | None = None         # filename-parsing mode (fallback)
    conditions: dict[str, str] = field(default_factory=lambda: {"age": "numeric"})
    image_size: tuple[int, int, int] = (224, 224, 224)
    val_split: float = 0.2
    num_workers: int = 8
    # CSV-only options
    filepath_col: str = "FilePath"
    subject_id_col: str = "SubjectID"
    filter_col: str | None = "Condition"
    filter_value: str | None = "CN"
    # filename-mode option
    require_conditions_in_filename: bool = True

    def __post_init__(self):
        self.image_size = tuple(int(v) for v in self.image_size)
        if len(self.image_size) != 3:
            raise ValueError(f"data.image_size must have 3 values, got {self.image_size}")
        self.conditions = {str(k).lower(): str(v).lower() for k, v in self.conditions.items()}
        for name, kind in self.conditions.items():
            if kind not in ("numeric", "categorical"):
                raise ValueError(f"condition '{name}' has invalid type '{kind}' (numeric|categorical)")


@dataclass
class ModelConfig:
    """HDiT backbone geometry and the flow-matching objective."""

    patch_size: tuple[int, int] = (8, 8)
    cond_embed_dim: int = 256
    slice_embed_dim: int = 256
    flow: str = "cfm"
    morpheus_scale: float = 1.0
    levels: list[dict] = field(default_factory=list)
    mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        self.patch_size = tuple(int(v) for v in self.patch_size)
        if len(self.patch_size) != 2:
            raise ValueError(f"model.patch_size must have 2 values, got {self.patch_size}")
        if self.flow not in FLOW_FORMULATIONS:
            raise ValueError(f"model.flow must be one of {FLOW_FORMULATIONS}, got '{self.flow}'")


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 200
    num_flow_steps_sampling: int = 10
    resume_from: str | None = None


@dataclass
class SamplingConfig:
    sampler: str = "heun"
    cfg_rescale: float = 0.7

    def __post_init__(self):
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampling.sampler must be one of {SAMPLERS}, got '{self.sampler}'")


@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "WaveDiT"
    wandb_entity: str | None = None
    checkpoint_dir: str = "./checkpoints"


@dataclass
class PostTrainGenConfig:
    """Optional sample generation immediately after training finishes."""

    enabled: bool = False
    num_samples: int = 2
    conditions: list[str] = field(default_factory=lambda: ["age=35.0", "age=75.0"])
    save_size: tuple[int, int, int] = (182, 218, 182)
    cfg_scale: float = 1.0

    def __post_init__(self):
        self.save_size = tuple(int(v) for v in self.save_size)


@dataclass
class Config:
    run_name: str = "wavedit_run"
    seed: int = 42
    device: str = "cuda"
    precision: str = "bf16"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    post_train_generation: PostTrainGenConfig = field(default_factory=PostTrainGenConfig)

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got '{self.precision}'")

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        section_types = {
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "sampling": SamplingConfig,
            "logging": LoggingConfig,
            "post_train_generation": PostTrainGenConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in section_types:
                kwargs[key] = _build_section(section_types[key], value)
            elif key in {f.name for f in fields(cls)}:
                kwargs[key] = value
            else:
                logger.warning("Ignoring unknown top-level config key: '%s'", key)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded configuration from %s", path)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_section(section_cls, value: dict[str, Any] | None):
    if value is None:
        return section_cls()
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping for section '{section_cls.__name__}', got {type(value)}")
    known = {f.name for f in fields(section_cls)}
    kwargs = {}
    for key, val in value.items():
        if key in known:
            kwargs[key] = val
        else:
            logger.warning("Ignoring unknown key '%s' in section '%s'", key, section_cls.__name__)
    return section_cls(**kwargs)

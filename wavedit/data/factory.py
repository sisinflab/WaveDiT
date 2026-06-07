"""Turn a :class:`~wavedit.config.DataConfig` into ready-to-use datasets plus the
condition metadata that the model and sampler need.

The condition-metadata helpers (:func:`build_condition_config`,
:func:`build_null_conditions`) are also reused at generation time to reconstruct
the same configuration from a checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from ..config import DataConfig
from ..utils import get_logger
from .dataset import BrainMRIDataset
from .transforms import build_train_transform, split_dataset

logger = get_logger(__name__)


@dataclass
class DatasetBundle:
    train: Dataset
    val: Dataset | None
    condition_config: dict[str, dict[str, Any]]
    condition_ranges: dict[str, dict[str, float]]
    categorical_maps: dict[str, dict]
    cardinalities: dict[str, int]
    null_conditions: dict[str, float]


def build_condition_config(condition_types: dict[str, str], cardinalities: dict[str, int]) -> dict[str, dict]:
    """Build the ``{name: {'type', ['num_categories']}}`` spec the backbone expects."""
    config = {}
    for name, kind in condition_types.items():
        spec: dict[str, Any] = {"type": kind}
        if kind == "categorical":
            if name not in cardinalities:
                raise ValueError(f"Missing cardinality for categorical condition '{name}'.")
            spec["num_categories"] = cardinalities[name]
        config[name] = spec
    return config


def build_null_conditions(condition_config: dict[str, dict], condition_ranges: dict[str, dict]) -> dict[str, float]:
    """Raw 'null' condition values used as the unconditional branch for CFG.

    Numeric -> midpoint of the training range (normalises to 0.5).
    Categorical -> the dedicated null class id (``num_categories``); the backbone
    embedding reserves one extra row for it.
    """
    nulls = {}
    for name, spec in condition_config.items():
        if spec["type"] == "numeric":
            rng = condition_ranges.get(name)
            if rng and rng["max"] > rng["min"]:
                nulls[name] = (rng["min"] + rng["max"]) / 2.0
            elif rng:
                nulls[name] = rng["min"]
            else:
                nulls[name] = 0.5
        else:
            nulls[name] = float(spec["num_categories"])
    return nulls


def build_datasets(data_cfg: DataConfig, seed: int) -> DatasetBundle:
    if data_cfg.metadata_csv:
        full = BrainMRIDataset.from_csv(
            metadata_csv=data_cfg.metadata_csv,
            condition_types=data_cfg.conditions,
            image_size=data_cfg.image_size,
            filepath_col=data_cfg.filepath_col,
            filter_col=data_cfg.filter_col,
            filter_value=data_cfg.filter_value,
            transform=None,
        )
    elif data_cfg.data_folder:
        full = BrainMRIDataset.from_folder(
            data_folder=data_cfg.data_folder,
            condition_types=data_cfg.conditions,
            image_size=data_cfg.image_size,
            require_conditions=data_cfg.require_conditions_in_filename,
            transform=None,
        )
    else:
        raise ValueError("data.metadata_csv or data.data_folder must be set.")

    train, val = split_dataset(full, data_cfg.val_split, seed, build_train_transform(), None)

    cardinalities = {name: len(m) for name, m in full.categorical_maps.items()}
    condition_config = build_condition_config(data_cfg.conditions, cardinalities)
    null_conditions = build_null_conditions(condition_config, full.condition_ranges)

    return DatasetBundle(
        train=train,
        val=val,
        condition_config=condition_config,
        condition_ranges=full.condition_ranges,
        categorical_maps=full.categorical_maps,
        cardinalities=cardinalities,
        null_conditions=null_conditions,
    )

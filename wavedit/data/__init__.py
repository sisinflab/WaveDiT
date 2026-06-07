"""Data loading: a unified dataset, augmentation, collation and a config-driven factory."""

from .collate import collate_fn
from .dataset import AGE_PATTERN, BrainMRIDataset
from .factory import (
    DatasetBundle,
    build_condition_config,
    build_datasets,
    build_null_conditions,
)
from .transforms import TransformedSubset, build_train_transform, split_dataset

__all__ = [
    "AGE_PATTERN",
    "BrainMRIDataset",
    "DatasetBundle",
    "build_condition_config",
    "build_datasets",
    "build_null_conditions",
    "TransformedSubset",
    "build_train_transform",
    "split_dataset",
    "collate_fn",
]

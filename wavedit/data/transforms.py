"""Training-time augmentations and the train/validation split utilities."""

from __future__ import annotations

import numpy as np
import torch
from monai.transforms import Compose, RandGaussianNoise, RandRotate, RandScaleIntensity
from torch.utils.data import Dataset, Subset, random_split

from ..utils import get_logger

logger = get_logger(__name__)


def build_train_transform():
    """Light geometric/intensity augmentation applied only to the training split."""
    return Compose([
        RandRotate(range_x=np.pi / 36, range_y=np.pi / 36, range_z=np.pi / 36,
                   prob=0.4, keep_size=True, mode="trilinear"),
        RandScaleIntensity(factors=0.2, prob=0.2),
        RandGaussianNoise(std=0.02, prob=0.2),
    ])


class TransformedSubset(Dataset):
    """A view over a ``Subset`` that applies its own transform.

    The base dataset stores a single ``transform`` attribute; this wrapper swaps it
    in for the duration of each access so the train and validation splits can use
    different augmentations while sharing one underlying dataset.
    """

    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        base = subset
        while isinstance(base, Subset):
            base = base.dataset
        self.base_dataset = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index):
        original_index = self.subset.indices[index]
        previous = self.base_dataset.transform
        self.base_dataset.transform = self.transform
        try:
            return self.base_dataset[original_index]
        finally:
            self.base_dataset.transform = previous


def split_dataset(full_dataset, val_split: float, seed: int, train_transform, val_transform=None):
    """Split ``full_dataset`` into (train, val) ``TransformedSubset`` views.

    ``val`` is ``None`` when ``val_split <= 0``.
    """
    n = len(full_dataset)
    if val_split <= 0.0:
        train = TransformedSubset(Subset(full_dataset, list(range(n))), train_transform)
        logger.info("No validation split; training on all %d samples.", n)
        return train, None
    if val_split >= 1.0:
        raise ValueError("val_split must be < 1.0 to leave samples for training.")

    train_size = int((1.0 - val_split) * n)
    val_size = n - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Split produced empty set (train={train_size}, val={val_size}); adjust val_split.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    logger.info("Split dataset: %d train / %d val.", train_size, val_size)
    return TransformedSubset(train_subset, train_transform), TransformedSubset(val_subset, val_transform)

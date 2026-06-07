"""Batch collation that tolerates samples dropped by the dataset (returned ``None``)."""

from __future__ import annotations

import torch

from ..utils import get_logger

logger = get_logger(__name__)


def collate_fn(batch):
    """Collate ``(image, conditions)`` samples, skipping any ``None`` entries.

    Returns ``(images, conditions)`` where ``images`` is ``(B, 1, D, H, W)`` and
    ``conditions`` maps each name to a ``(B, 1)`` tensor, or ``None`` if the whole
    batch was dropped.
    """
    valid = [sample for sample in batch if sample is not None]
    if not valid:
        logger.warning("Entire batch failed to load; skipping.")
        return None

    images = torch.stack([sample[0] for sample in valid], dim=0)

    condition_dicts = [sample[1] for sample in valid]
    conditions = {}
    for key in condition_dicts[0]:
        if all(key in d for d in condition_dicts):
            conditions[key] = torch.stack([d[key] for d in condition_dicts], dim=0)
        else:
            logger.warning("Condition '%s' missing from some samples; dropping it for this batch.", key)
    return images, conditions

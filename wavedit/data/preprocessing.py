"""Image preprocessing shared by all dataset modes: intensity normalisation and
spatial padding to the model input size."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_logger

logger = get_logger(__name__)

_CLIP_PERCENTILES = (0.5, 99.5)


def robust_normalize(volume: np.ndarray) -> np.ndarray:
    """Percentile-clip a volume and rescale it to ``[-1, 1]`` (float32).

    Uses the 0.5/99.5 percentiles to be robust to MRI intensity outliers. Returns
    zeros if the dynamic range collapses.
    """
    p_low, p_high = np.percentile(volume, _CLIP_PERCENTILES)
    denom = p_high - p_low
    if denom < 1e-8:
        logger.warning("Normalisation range collapsed (denom=%.3e); returning zeros.", denom)
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / denom
    volume = volume * 2.0 - 1.0
    if not np.isfinite(volume).all():
        logger.error("Non-finite values after normalisation; forcing finite.")
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=-1.0)
    return volume.astype(np.float32)


def pad_to_size(volume: torch.Tensor, target_size: tuple[int, int, int]) -> torch.Tensor:
    """Symmetrically replicate-pad the last three (spatial) dims up to ``target_size``.

    Accepts a 3D ``(D, H, W)``, 4D ``(C, D, H, W)`` or 5D ``(N, C, D, H, W)`` tensor
    and returns the same rank.
    """
    current = volume.shape[-3:]
    pad_per_dim = [max(0, t - c) for t, c in zip(target_size, current)]
    if all(p == 0 for p in pad_per_dim):
        return volume

    # F.pad expects padding ordered from the last dim backwards: (W, H, D).
    pad_d, pad_h, pad_w = pad_per_dim
    padding = (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_d // 2, pad_d - pad_d // 2,
    )

    original_ndim = volume.ndim
    if original_ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 4:
        volume = volume.unsqueeze(0)
    elif original_ndim != 5:
        raise ValueError(f"pad_to_size expects a 3D/4D/5D tensor, got {original_ndim}D")

    padded = F.pad(volume, padding, mode="replicate")

    if original_ndim == 3:
        padded = padded.squeeze(0).squeeze(0)
    elif original_ndim == 4:
        padded = padded.squeeze(0)

    if tuple(padded.shape[-3:]) != tuple(target_size):
        logger.warning("Padding produced shape %s, expected spatial %s", tuple(padded.shape[-3:]), tuple(target_size))
    return padded

"""Generate and save conditioned brain-MRI volumes with a trained WaveDiT model."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..utils import get_logger

logger = get_logger(__name__)


def center_crop_bounds(full_size: tuple[int, int, int], target_size: tuple[int, int, int]):
    """Return per-axis ``(start, end)`` slices to center-crop ``full_size`` to ``target_size``."""
    if any(t > f for t, f in zip(target_size, full_size)):
        raise ValueError(f"Cannot crop {full_size} down to a larger {target_size}.")
    bounds = []
    for full, target in zip(full_size, target_size):
        start = (full - target) // 2
        bounds.append((start, start + target))
    return bounds


def parse_condition_sets(condition_strings: list[str], condition_config: dict) -> list[dict]:
    """Parse ``"age=35.0 sex=1"``-style strings into raw-condition dicts.

    Numeric conditions become floats; categorical conditions become ints when
    possible, otherwise the raw string (mapped to a class id at sampling time).
    Keys not present in ``condition_config`` are ignored with a warning.
    """
    parsed_sets = []
    for condition_string in condition_strings:
        current = {}
        for pair in condition_string.split():
            if "=" not in pair:
                logger.warning("Ignoring malformed condition token '%s' (expected key=value).", pair)
                continue
            key, value = pair.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key not in condition_config:
                logger.warning("Condition '%s' is not used by the model; ignoring.", key)
                continue
            if condition_config[key]["type"] == "numeric":
                current[key] = float(value)
            else:
                try:
                    current[key] = int(value)
                except ValueError:
                    current[key] = value
        if current:
            parsed_sets.append(current)
    return parsed_sets


def _condition_tag(conditions: dict) -> str:
    parts = []
    for key, value in sorted(conditions.items()):
        value_str = f"{value:.2f}" if isinstance(value, float) else str(value).replace(" ", "_")
        parts.append(f"{key}_{value_str}")
    return "_".join(parts) if parts else "unconditional"


@torch.no_grad()
def generate_samples(
    model,
    conditions: list[dict],
    num_samples_per_condition: int,
    output_dir: str | Path,
    save_size: tuple[int, int, int],
    model_output_size: tuple[int, int, int],
    cfg_scale: float = 1.0,
    sampler: str = "heun",
    morpheus_scale: float | None = None,
    cfg_rescale: float = 0.7,
    group_by_condition: bool = True,
    filename_prefix: str = "WaveDiT",
    autocast_dtype: "torch.dtype | None" = None,
) -> None:
    """Generate ``num_samples_per_condition`` volumes for each raw-condition dict.

    Saved volumes are rescaled from the model's ``[-1, 1]`` range to ``[0, 1]`` and
    center-cropped to ``save_size``. With ``group_by_condition`` each condition gets
    its own sub-directory; otherwise everything is written flat (useful for long
    interpolation sweeps).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    amp_dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32

    crop_bounds = None
    if tuple(save_size) != tuple(model_output_size):
        crop_bounds = center_crop_bounds(tuple(model_output_size), tuple(save_size))
        logger.info("Cropping generated volumes from %s to %s.", model_output_size, save_size)

    for raw_conditions in conditions:
        tag = _condition_tag(raw_conditions)
        target_dir = output_dir / tag if group_by_condition else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            volumes = model.sample(
                num_samples=num_samples_per_condition,
                raw_conditions=raw_conditions,
                cfg_scale=cfg_scale,
                sampler=sampler,
                morpheus_scale=morpheus_scale,
                cfg_rescale=cfg_rescale,
                autocast_dtype=amp_dtype,
            )

        if crop_bounds is not None:
            (d0, d1), (h0, h1), (w0, w1) = crop_bounds
            volumes = volumes[:, :, d0:d1, h0:h1, w0:w1]

        volumes = torch.clamp((volumes.float() + 1.0) / 2.0, 0.0, 1.0)
        logger.info("Saving %d sample(s) for [%s] -> %s", num_samples_per_condition, tag, target_dir)

        for i in range(num_samples_per_condition):
            data = volumes[i, 0].cpu().numpy().astype(np.float32)
            nifti = nib.Nifti1Image(data, affine=np.eye(4))
            filename = f"{filename_prefix}_{tag}_sample{i:03d}.nii.gz"
            nib.save(nifti, os.path.join(target_dir, filename))

    logger.info("Finished generating samples in %s", output_dir)

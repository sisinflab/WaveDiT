"""Validation-time visualisation logged to Weights & Biases.

All condition metadata is read directly off the model (``condition_config``,
``condition_ranges``, ``categorical_maps``, ``null_conditions``), so callers only
need to supply the model, a data sample and the output size.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from ..utils import get_logger
from ..wavelets import dwt_3d, idwt_3d
from .metrics import evaluate_recon_quality

logger = get_logger(__name__)


def create_ortho_view(volume: np.ndarray) -> np.ndarray:
    """Compose central axial/coronal/sagittal slices of a 3D volume into one RGB image."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    d, h, w = volume.shape
    d_mid, h_mid, w_mid = d // 2, h // 2, w // 2

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={"width_ratios": [w, h], "height_ratios": [d, h]})
    axs[0, 0].imshow(volume[:, h_mid, :].T, cmap="gray", origin="lower")
    axs[0, 0].set_title(f"Coronal (Y={h_mid})")
    axs[0, 1].imshow(np.rot90(volume[:, :, w_mid], k=1), cmap="gray")
    axs[0, 1].set_title(f"Sagittal (X={w_mid})")
    axs[1, 0].imshow(volume[d_mid, :, :], cmap="gray", origin="lower")
    axs[1, 0].set_title(f"Axial (Z={d_mid})")
    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout(pad=0.5)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb


def _dwt_roundtrip(image: torch.Tensor) -> torch.Tensor:
    """Reconstruct an image through the wavelet latent (shows the latent's fidelity)."""
    coeffs = torch.cat(dwt_3d(image), dim=1)
    lll = coeffs[:, :1, ...]
    hf = torch.split(coeffs[:, 1:, ...], 1, dim=1)
    return torch.clamp(idwt_3d(lll, *hf), -1.0, 1.0)


def _to_unit_interval(volume: torch.Tensor) -> np.ndarray:
    return np.clip((volume.cpu().numpy() + 1.0) / 2.0, 0.0, 1.0)


@torch.no_grad()
def visualize_generation(model, val_loader, model_output_size, *,
                         sampler="heun", cfg_scale=1.0, cfg_rescale=0.7, epoch=None, use_wandb=True,
                         autocast_dtype=None):
    """Log an orthogonal view of a real validation volume next to a generated one."""
    device = next(model.parameters()).device
    amp_dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    model.eval()
    try:
        batch = next(iter(val_loader))
        if batch is None:
            logger.warning("visualize_generation: empty batch; skipping.")
            return
        images, _ = batch
        real = images[:1].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            real_recon = _dwt_roundtrip(real)
            synthetic = model.sample(
                num_samples=1, raw_conditions=dict(model.null_conditions),
                cfg_scale=cfg_scale, sampler=sampler, cfg_rescale=cfg_rescale,
                autocast_dtype=amp_dtype,
            )

        psnr_val, ssim_val = evaluate_recon_quality(real_recon, synthetic)
        suffix = f"_Ep{epoch}" if epoch is not None else "_Final"

        if use_wandb:
            wandb.log({
                f"generation/real{suffix}": wandb.Image(
                    create_ortho_view(_to_unit_interval(real_recon[0, 0])), caption="Real (wavelet round-trip)"),
                f"generation/synthetic{suffix}": wandb.Image(
                    create_ortho_view(_to_unit_interval(synthetic[0, 0])),
                    caption=f"WaveDiT (PSNR {psnr_val:.2f}, SSIM {ssim_val:.4f}, CFG {cfg_scale:.1f})"),
            })
        else:
            logger.info("Generation viz (epoch %s): PSNR=%.2f SSIM=%.4f", epoch, psnr_val, ssim_val)
    except StopIteration:
        logger.warning("visualize_generation: validation loader exhausted.")
    except Exception as exc:  # noqa: BLE001
        logger.error("visualize_generation failed: %s", exc, exc_info=True)


def _sweep_values(model, condition_key, num_values):
    spec = model.condition_config[condition_key]
    if spec["type"] == "numeric":
        rng = model.condition_ranges.get(condition_key, {"min": 0.0, "max": 1.0})
        lo, hi = rng["min"], rng["max"]
        if num_values <= 1 or hi <= lo:
            return [(lo + hi) / 2.0]
        return torch.linspace(lo, hi, num_values).tolist()
    categories = list(model.categorical_maps.get(condition_key, {}).keys())
    if not categories:
        return list(range(min(num_values, spec.get("num_categories", 1))))
    return categories[:num_values]


@torch.no_grad()
def visualize_condition_sweep(model, model_output_size, condition_key, *,
                              num_values=1, cfg_scale=1.5, epoch=None, use_wandb=True,
                              autocast_dtype=None):
    """Vary one condition (others held at their null values) and log a mid-slice for each value."""
    device = next(model.parameters()).device
    amp_dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    model.eval()
    condition_key = condition_key.lower()
    if condition_key not in model.condition_config:
        logger.warning("Condition '%s' not in model; skipping sweep.", condition_key)
        return

    suffix = f"_Ep{epoch}" if epoch is not None else "_Final"
    images_to_log = {}
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        for value in _sweep_values(model, condition_key, num_values):
            raw_conditions = dict(model.null_conditions)
            raw_conditions[condition_key] = value
            volume = model.sample(num_samples=1, raw_conditions=raw_conditions, cfg_scale=cfg_scale,
                                  autocast_dtype=amp_dtype)
            mid = volume.shape[2] // 2
            slice_img = _to_unit_interval(volume[0, 0, mid])
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            key = f"sweep_{condition_key}/{value_str}{suffix}".replace(".", "_")
            images_to_log[key] = wandb.Image(slice_img, caption=f"{condition_key}={value_str} (CFG {cfg_scale:.1f})")

    if use_wandb and images_to_log:
        wandb.log(images_to_log)
        logger.info("Logged condition sweep over '%s'.", condition_key)

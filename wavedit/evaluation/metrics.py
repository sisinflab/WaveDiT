"""Reconstruction-quality metrics (PSNR / SSIM) for sanity visualisation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from ..utils import get_logger

logger = get_logger(__name__)


def evaluate_recon_quality(real_images, generated_images) -> tuple[float, float]:
    """Mean PSNR/SSIM over a batch of 3D volumes, each rescaled from [-1, 1] to [0, 1]."""
    real = (real_images.detach().float().cpu().numpy() + 1.0) / 2.0
    gen = (generated_images.detach().float().cpu().numpy() + 1.0) / 2.0
    if real.ndim == 5 and real.shape[1] == 1:
        real = real.squeeze(1)
    if gen.ndim == 5 and gen.shape[1] == 1:
        gen = gen.squeeze(1)
    if real.ndim != 4 or gen.ndim != 4:
        logger.warning("Unexpected metric shapes real=%s gen=%s; expected 4D.", real.shape, gen.shape)
        return 0.0, 0.0

    psnr_values, ssim_values = [], []
    for i in range(real.shape[0]):
        try:
            psnr_values.append(psnr(real[i], gen[i], data_range=1.0))
            win_size = min(7, *real[i].shape)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size < 3:
                logger.warning("Skipping SSIM for sample %d (dims too small: %s).", i, real[i].shape)
                ssim_values.append(0.0)
                continue
            ssim_values.append(ssim(real[i], gen[i], data_range=1.0, channel_axis=None,
                                    win_size=win_size, gaussian_weights=True))
        except Exception as exc:  # noqa: BLE001
            logger.error("PSNR/SSIM failed for sample %d: %s", i, exc)
            psnr_values.append(0.0)
            ssim_values.append(0.0)

    return (float(np.mean(psnr_values)) if psnr_values else 0.0,
            float(np.mean(ssim_values)) if ssim_values else 0.0)

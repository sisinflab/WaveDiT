"""Build a :class:`WaveletFlowMatching` model from a :class:`~wavedit.config.Config`.

Used by both training and generation so the architecture is derived from the config
in exactly one place. The wavelet latent is always 8 channels (one Haar level) at
half the input spatial resolution.
"""

from __future__ import annotations

from ..config import Config
from ..utils import get_logger
from .wavelet_flow_matching import WaveletFlowMatching

logger = get_logger(__name__)

LATENT_CHANNELS = 8  # one 3D Haar level: 1 LLL + 7 HF bands


def build_model(
    cfg: Config,
    condition_config: dict,
    condition_ranges: dict,
    categorical_maps: dict,
    null_conditions: dict,
    num_flow_steps: int | None = None,
) -> WaveletFlowMatching:
    if not cfg.model.levels:
        raise ValueError("model.levels is empty; define at least one HDiT level in the config.")
    if not cfg.model.mapping:
        raise ValueError("model.mapping is empty; define the mapping network in the config.")

    d_coeff, h_coeff, w_coeff = (s // 2 for s in cfg.data.image_size)
    patch_h, patch_w = cfg.model.patch_size
    if h_coeff % patch_h != 0 or w_coeff % patch_w != 0:
        raise ValueError(
            f"Wavelet slice size ({h_coeff}, {w_coeff}) must be divisible by patch_size "
            f"{cfg.model.patch_size}. Adjust data.image_size or model.patch_size."
        )

    return WaveletFlowMatching(
        latent_channels=LATENT_CHANNELS,
        patch_size=cfg.model.patch_size,
        levels_config=cfg.model.levels,
        mapping_config=cfg.model.mapping,
        condition_config=condition_config,
        cond_embed_dim=cfg.model.cond_embed_dim,
        num_slices=d_coeff,
        slice_embed_dim=cfg.model.slice_embed_dim,
        latent_shape=(LATENT_CHANNELS, d_coeff, h_coeff, w_coeff),
        num_flow_steps_sampling=num_flow_steps or cfg.train.num_flow_steps_sampling,
        condition_ranges=condition_ranges,
        categorical_maps=categorical_maps,
        null_conditions=null_conditions,
        flow_formulation=cfg.model.flow,
        morpheus_scale=cfg.model.morpheus_scale,
    )

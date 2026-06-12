"""3D wavelet backbone built on the HDiT transformer.

A 3D wavelet volume ``(B, C_in, D, H, W)`` is processed slice-wise: the depth axis
``D`` is folded into the batch so each axial slice is a 2D image for the HDiT
transformer, while a depth argument (``d_c``) lets the spatio-temporal attention
blocks attend across slices. Conditioning (subject metadata + slice index +
Morpheus frequency hint) is fed through the transformer's mapping network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import get_logger
from .hdit import (
    FourierFeatures,
    GlobalAttentionSpec,
    ImageTransformerDenoiserModelV2,
    LevelSpec,
    MappingSpec,
    NeighborhoodAttentionSpec,
    ShiftedWindowAttentionSpec,
    SpatioTemporalAttentionSpec,
)

logger = get_logger(__name__)

_ATTENTION_SPECS = {
    "global": (GlobalAttentionSpec, {"d_head"}),
    "spatio-temporal": (SpatioTemporalAttentionSpec, {"d_head"}),
    "neighborhood": (NeighborhoodAttentionSpec, {"d_head", "kernel_size"}),
    "shifted-window": (ShiftedWindowAttentionSpec, {"d_head", "window_size"}),
}


def _build_attention_spec(attn_conf: dict):
    attn_type = attn_conf.get("type")
    if attn_type not in _ATTENTION_SPECS:
        raise ValueError(f"Unsupported self_attn type '{attn_type}'. Choose from {sorted(_ATTENTION_SPECS)}.")
    spec_cls, allowed = _ATTENTION_SPECS[attn_type]
    unknown = set(attn_conf) - {"type"} - allowed
    if unknown:
        raise ValueError(f"Unsupported keys for {attn_type} attention: {sorted(unknown)}")
    kwargs = {k: attn_conf[k] for k in allowed if k in attn_conf}
    return spec_cls(**kwargs)


def _build_level_specs(levels_config: list[dict]) -> list[LevelSpec]:
    specs = []
    for level in levels_config:
        width = level["width"]
        specs.append(LevelSpec(
            depth=level["depth"],
            width=width,
            d_ff=level.get("d_ff", width * 3),
            self_attn=_build_attention_spec(level["self_attn"]),
            dropout=level.get("dropout", 0.0),
        ))
    return specs


def _build_mapping_spec(mapping_config: dict) -> MappingSpec:
    width = mapping_config["width"]
    return MappingSpec(
        depth=mapping_config["depth"],
        width=width,
        d_ff=mapping_config.get("d_ff", width * 3),
        dropout=mapping_config.get("dropout", 0.0),
    )


class DiT3DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int],
        levels_config: list[dict],
        mapping_config: dict,
        condition_config: dict[str, dict],
        cond_embed_dim: int,
        num_slices: int,
        slice_embed_dim: int,
        latent_shape: tuple[int, int, int, int],
        initial_slice_hw: tuple[int, int],
        freq_hint_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_config = condition_config or {}
        self.cond_embed_dim = cond_embed_dim
        self.num_slices = num_slices
        self.slice_embed_dim = slice_embed_dim
        self.latent_shape = latent_shape  # (out_channels, D, H, W) — used by the sampler
        self.freq_hint_dim = freq_hint_dim

        # Per-condition embedders. Categorical embeddings reserve one extra row as a
        # dedicated null token for classifier-free guidance.
        self.condition_embedders = nn.ModuleDict()
        for name, spec in self.condition_config.items():
            if spec["type"] == "numeric":
                self.condition_embedders[name] = nn.Linear(1, cond_embed_dim)
            elif spec["type"] == "categorical":
                num_categories = spec["num_categories"]
                self.condition_embedders[name] = nn.Embedding(num_categories + 1, cond_embed_dim)
            else:
                raise ValueError(f"Unknown condition type for '{name}': {spec['type']}")
        total_cond_dim = len(self.condition_config) * cond_embed_dim

        self.slice_idx_embedder = FourierFeatures(1, slice_embed_dim)

        self.freq_hint_projector = None
        if freq_hint_dim > 0:
            self.freq_hint_projector = nn.Linear(freq_hint_dim, slice_embed_dim)

        extra = slice_embed_dim if freq_hint_dim > 0 else 0
        self.mapping_cond_dim = total_cond_dim + slice_embed_dim + extra

        self.transformer = ImageTransformerDenoiserModelV2(
            levels=_build_level_specs(levels_config),
            mapping=_build_mapping_spec(mapping_config),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=tuple(patch_size),
            num_classes=0,
            mapping_cond_dim=self.mapping_cond_dim,
            initial_slice_hw=initial_slice_hw,
        )
        logger.info(
            "DiT3DBackbone: in=%d out=%d patch=%s conds=%s mapping_cond_dim=%d",
            in_channels, out_channels, tuple(patch_size), list(self.condition_config), self.mapping_cond_dim,
        )

    def _embed_conditions(self, conditions: dict, batch_size: int, device, dtype) -> torch.Tensor | None:
        if not self.condition_config:
            return None
        parts = []
        for name, spec in self.condition_config.items():
            value = conditions.get(name)
            if value is None:
                logger.warning("Condition '%s' missing at forward; using zeros.", name)
                parts.append(torch.zeros(batch_size, self.cond_embed_dim, device=device, dtype=dtype))
                continue
            embedder = self.condition_embedders[name]
            if spec["type"] == "categorical":
                value = value.squeeze(-1).long() if value.ndim > 1 else value.long()
            parts.append(embedder(value))
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, conditions: dict, freq_hint: torch.Tensor | None = None):
        batch, channels, depth, height, width = x.shape
        device, dtype = x.device, x.dtype

        sigma = torch.clamp(time, min=1e-5)

        # Fold depth into the batch: (B, C, D, H, W) -> (B*D, C, H, W).
        x_slices = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        sigma_slices = sigma.unsqueeze(1).repeat(1, depth).reshape(batch * depth)

        # Normalised slice index in [-1, 1] -> Fourier features.
        slice_index = torch.arange(depth, device=device, dtype=dtype).repeat(batch)
        if self.num_slices > 1:
            slice_index = (slice_index / (self.num_slices - 1.0)) * 2.0 - 1.0
        else:
            slice_index = torch.zeros_like(slice_index)
        slice_emb = self.slice_idx_embedder(slice_index.unsqueeze(-1))

        parts = []
        cond_emb = self._embed_conditions(conditions, batch, device, dtype)
        if cond_emb is not None:
            parts.append(cond_emb.unsqueeze(1).repeat(1, depth, 1).reshape(batch * depth, -1))
        parts.append(slice_emb)
        if self.freq_hint_projector is not None:
            if freq_hint is None:
                freq_hint = torch.zeros(batch, self.freq_hint_dim, device=device, dtype=dtype)
            hint_emb = self.freq_hint_projector(freq_hint)
            parts.append(hint_emb.unsqueeze(1).repeat(1, depth, 1).reshape(batch * depth, -1))

        mapping_cond = torch.cat(parts, dim=1) if parts else None
        if mapping_cond is not None and mapping_cond.shape[1] != self.mapping_cond_dim:
            raise ValueError(f"mapping_cond dim {mapping_cond.shape[1]} != expected {self.mapping_cond_dim}")

        velocity_slices = self.transformer(x_slices, sigma_slices, mapping_cond=mapping_cond, d_c=depth)

        # Unfold depth back out: (B*D, C_out, H, W) -> (B, C_out, D, H, W).
        return velocity_slices.reshape(batch, depth, self.out_channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

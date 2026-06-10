"""WaveletFlowMatching: the top-level WaveDiT model.

Ties together the wavelet transform, the Morpheus uncertainty scheduler and the
HDiT backbone into a conditional flow-matching generator that operates in the 3D
Haar wavelet domain.

Pipeline
--------
* Training: the target image is transformed to its 8-channel wavelet latent; a flow
  path (``rectified`` | ``cfm`` | ``ot_fm``) defines the interpolant ``x_t`` and the
  target velocity. The latent is augmented with an energy channel (8 -> 9), Morpheus
  predicts per-channel log-variances, and the backbone predicts the velocity. The
  loss is a Bayesian heteroscedastic-weighted velocity MSE.
* Sampling: integrate the probability-flow ODE from noise, optionally with
  classifier-free guidance and Morpheus uncertainty-minimisation guidance, then
  invert the wavelet transform back to image space.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..utils import get_logger
from ..wavelets import dwt_3d, idwt_3d
from .backbone import DiT3DBackbone
from .sampling import integrate_ode
from .uncertainty import StateAwareUncertaintyScheduler

try:
    from flow_matching.path import AffineProbPath
    from flow_matching.path.scheduler import CondOTScheduler
    OT_FM_AVAILABLE = True
except ImportError:
    AffineProbPath = CondOTScheduler = None
    OT_FM_AVAILABLE = False

logger = get_logger(__name__)

NUM_LLL_CHANNELS = 1  # Haar: 1 low-frequency band + 7 high-frequency bands
LOG_INTERVAL = 100


class WaveletFlowMatching(nn.Module):
    def __init__(
        self,
        latent_channels: int = 8,
        patch_size: tuple[int, int] = (8, 8),
        levels_config: list[dict] | None = None,
        mapping_config: dict | None = None,
        condition_config: dict[str, dict] | None = None,
        cond_embed_dim: int = 256,
        num_slices: int = 112,
        slice_embed_dim: int = 256,
        latent_shape: tuple[int, int, int, int] = (8, 112, 112, 112),
        num_flow_steps_sampling: int = 100,
        condition_ranges: dict | None = None,
        categorical_maps: dict | None = None,
        null_conditions: dict | None = None,
        flow_formulation: str = "rectified",
        morpheus_scale: float = 1.0,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.num_flow_steps_sampling = num_flow_steps_sampling
        self.flow_formulation = flow_formulation.lower()
        self.morpheus_scale = morpheus_scale

        self.condition_config = condition_config or {}
        self.condition_ranges = condition_ranges or {}
        self.categorical_maps = categorical_maps or {}
        self.null_conditions = null_conditions or {}

        if self.flow_formulation not in ("rectified", "cfm", "ot_fm"):
            raise ValueError(f"Unknown flow_formulation '{self.flow_formulation}'.")
        self.ot_path = None
        if self.flow_formulation == "ot_fm":
            if not OT_FM_AVAILABLE:
                raise ImportError("flow_formulation='ot_fm' requires the 'flow-matching' package.")
            self.ot_path = AffineProbPath(scheduler=CondOTScheduler())

        # Morpheus sees the 8 wavelet channels + 1 energy channel and predicts a
        # log-variance per target channel.
        self.uncertainty = StateAwareUncertaintyScheduler(
            num_target_channels=latent_channels,
            input_channels=latent_channels + 1,
            time_embed_dim=64,
        )

        self.backbone = DiT3DBackbone(
            in_channels=latent_channels + 1,   # 8 wavelet + 1 energy
            out_channels=latent_channels,      # predict the 8-channel velocity
            patch_size=patch_size,
            levels_config=levels_config or [],
            mapping_config=mapping_config or {},
            condition_config=self.condition_config,
            cond_embed_dim=cond_embed_dim,
            num_slices=num_slices,
            slice_embed_dim=slice_embed_dim,
            latent_shape=latent_shape,
            initial_slice_hw=(latent_shape[2], latent_shape[3]),
            freq_hint_dim=latent_channels,
        )
        logger.info("WaveletFlowMatching ready (flow=%s, conditions=%s).",
                    self.flow_formulation, list(self.condition_config))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _augment_with_energy(self, latent: torch.Tensor) -> torch.Tensor:
        """Append a high-frequency energy channel: (B, 8, ...) -> (B, 9, ...)."""
        hf_bands = latent[:, NUM_LLL_CHANNELS:, ...]
        energy = torch.sqrt(torch.sum(hf_bands**2, dim=1, keepdim=True) + 1e-6)
        return torch.cat([latent, energy], dim=1)

    @staticmethod
    def _split_wavelet_coeffs(latent: torch.Tensor):
        lll = latent[:, :NUM_LLL_CHANNELS, ...]
        hf = torch.split(latent[:, NUM_LLL_CHANNELS:, ...], 1, dim=1)
        return (lll,) + tuple(hf)

    def _weighted_velocity_loss(self, v_pred, v_target, log_vars_map, global_step, use_wandb):
        """Bayesian heteroscedastic velocity loss: 0.5 * exp(-s) * MSE + 0.5 * s."""
        if not torch.isfinite(v_pred).all():
            logger.error("Step %s: non-finite model output.", global_step)
            return torch.tensor(float("inf"), device=v_pred.device), {}

        log_vars_map = torch.clamp(log_vars_map, min=-10.0, max=10.0)
        precision = torch.exp(-log_vars_map)
        loss_map = 0.5 * precision * (v_pred - v_target) ** 2 + 0.5 * log_vars_map
        loss = loss_map.mean()

        if not torch.isfinite(loss):
            logger.error("Step %s: non-finite velocity loss.", global_step)
            return torch.tensor(float("inf"), device=v_pred.device), {}

        diagnostics = {}
        if use_wandb and global_step % LOG_INTERVAL == 0:
            avg_precision = precision.mean(dim=0).flatten()
            for i, val in enumerate(avg_precision):
                diagnostics[f"uncertainty_weights/ch_{i}"] = val.item()
            lll_loss = loss_map[:, :NUM_LLL_CHANNELS, ...].mean()
            detail_loss = loss_map[:, NUM_LLL_CHANNELS:, ...].mean()
            diagnostics["loss/lll_component"] = lll_loss.item()
            diagnostics["loss/detail_component"] = detail_loss.item()
        return loss, diagnostics

    def _flow_targets(self, x1: torch.Tensor):
        """Sample a flow time and return ``(t, x_t, v_target)`` for the active formulation."""
        batch, device = x1.shape[0], x1.device
        x0 = torch.randn_like(x1)

        if self.flow_formulation == "rectified":
            # Bias time towards 0 (early, high-noise regime) with a power schedule.
            t = torch.rand(batch, device=device).pow(2.0)
            eps = 1e-4
            t = t * (1.0 - 2 * eps) + eps
            t_b = t.view(-1, *([1] * (x1.dim() - 1)))
            x_t = (1.0 - t_b) * x0 + t_b * x1
            return t, x_t, x1 - x0

        if self.flow_formulation == "cfm":
            t = torch.rand(batch, device=device)
            t_b = t.view(-1, *([1] * (x1.dim() - 1)))
            x_t = t_b * x1 + (1.0 - t_b) * x0
            return t, x_t, x1 - x0

        # ot_fm
        t = torch.rand(batch, device=device)
        sample = self.ot_path.sample(t=t, x_0=x0, x_1=x1)
        return t, sample.x_t, sample.dx_t

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def loss(self, images: torch.Tensor, conditions: dict, global_step: int = 0, use_wandb_logging: bool = False):
        x1 = torch.cat(dwt_3d(images), dim=1)  # (B, 8, D/2, H/2, W/2)
        t, x_t, v_target = self._flow_targets(x1)

        x_t_aug = self._augment_with_energy(x_t)
        log_vars_map, freq_hint = self.uncertainty(t, x_t_aug)
        v_pred = self.backbone(x_t_aug, t, conditions, freq_hint=freq_hint)

        loss, diagnostics = self._weighted_velocity_loss(v_pred, v_target, log_vars_map, global_step, use_wandb_logging)

        if use_wandb_logging and global_step % LOG_INTERVAL == 0:
            # No explicit step= : W&B silently drops rows whose step is behind the
            # internal counter once stepless logs (e.g. images) have advanced it.
            log_dict = {"loss/total_loss": loss.item(), "global_step": global_step, **diagnostics}
            wandb.log(log_dict)
        return loss

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _standardize_condition(self, name: str, raw_value, num_samples: int, device) -> torch.Tensor:
        spec = self.condition_config[name]
        if spec["type"] == "numeric":
            rng = self.condition_ranges.get(name, {"min": 0.0, "max": 1.0})
            span = rng["max"] - rng["min"]
            value = np.clip((float(raw_value) - rng["min"]) / span, 0.0, 1.0) if span > 1e-8 else 0.5
        else:  # categorical -> class id (falls back to the null token for unknown values)
            class_map = self.categorical_maps.get(name, {})
            value = float(class_map.get(str(raw_value), spec["num_categories"]))
        return torch.full((num_samples, 1), float(value), device=device, dtype=torch.float32)

    def _build_sampling_conditions(self, raw_conditions: dict, num_samples: int, device):
        cond, uncond = {}, {}
        for name in self.condition_config:
            raw = raw_conditions.get(name, self.null_conditions.get(name))
            cond[name] = self._standardize_condition(name, raw, num_samples, device)
            uncond[name] = self._standardize_condition(name, self.null_conditions.get(name), num_samples, device)
        return cond, uncond

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        raw_conditions: dict,
        cfg_scale: float = 1.0,
        sampler: str = "heun",
        morpheus_scale: float | None = None,
        cfg_rescale: float = 0.7,
        autocast_dtype: "torch.dtype | None" = None,
    ) -> torch.Tensor:
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        guidance_scale = morpheus_scale if morpheus_scale is not None else self.morpheus_scale

        # Match the training precision (default bf16) so inference does not silently
        # fall back to autocast's fp16 default, which under/overflows on the
        # heavy-tailed high-frequency wavelet bands.
        amp_dtype = autocast_dtype if autocast_dtype is not None else torch.bfloat16
        amp_enabled = device.type == "cuda" and amp_dtype != torch.float32

        cond, uncond = self._build_sampling_conditions(raw_conditions, num_samples, device)
        z0 = torch.randn((num_samples, *self.backbone.latent_shape), device=device)

        def velocity_fn(t_value: float, z: torch.Tensor) -> torch.Tensor:
            t = torch.full((num_samples,), t_value, device=device)

            # Morpheus guidance: descend the predicted uncertainty, scheduled to peak at t=0.5.
            dynamic_scale = guidance_scale * math.sin(math.pi * t_value)
            u_grad = torch.zeros_like(z)
            if dynamic_scale > 1e-4:
                with torch.enable_grad():
                    z = z.requires_grad_(True)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        log_vars, _ = self.uncertainty(t, self._augment_with_energy(z))
                        u_loss = log_vars.sum()
                    u_grad = torch.autograd.grad(u_loss, z)[0]
                z = z.detach()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                z_aug = self._augment_with_energy(z)
                _, hint = self.uncertainty(t, z_aug)
                v_cond = self.backbone(z_aug, t, cond, freq_hint=hint)

                if cfg_scale != 1.0:
                    v_uncond = self.backbone(z_aug, t, uncond, freq_hint=hint)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                    if cfg_rescale > 0.0:  # tame magnitude growth from strong guidance
                        std_cond = v_cond.std(dim=(1, 2, 3, 4), keepdim=True)
                        std_guided = v.std(dim=(1, 2, 3, 4), keepdim=True)
                        v = v * (std_cond / (std_guided + 1e-8)) * cfg_rescale + v * (1.0 - cfg_rescale)
                else:
                    v = v_cond

            return v - dynamic_scale * u_grad

        z_final = integrate_ode(velocity_fn, z0, self.num_flow_steps_sampling, sampler)
        image = idwt_3d(*self._split_wavelet_coeffs(z_final))
        return torch.clamp(image, -1.0, 1.0)

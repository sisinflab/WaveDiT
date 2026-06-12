"""Morpheus: the state-aware uncertainty scheduler.

Given the current (noised) wavelet state and the flow time ``t``, Morpheus predicts
a per-channel log-variance used to weight the velocity loss (Bayesian
heteroscedastic weighting) and to provide a frequency hint to the backbone.

Design notes:
* **Higher-order moments** — real wavelet coefficients are heavy-tailed while noise
  is Gaussian, so the state embedding includes skewness and kurtosis in addition to
  mean/std/max/L2 energy.
* **Soft bounding** — log-variances are squashed with ``10 * tanh(x / 10)`` to the
  range ``[-10, 10]`` so the ``exp`` term in the loss cannot explode or vanish.
* **Band-aware init** — the output bias starts at 0 for the high-energy LLL band and
  -2 for the seven low-energy high-frequency bands.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_STATS = 6  # mean, std, max, L2 energy, skewness, kurtosis
LOG_VAR_BOUND = 10.0


class StateAwareUncertaintyScheduler(nn.Module):
    def __init__(self, num_target_channels: int = 8, input_channels: int = 9, time_embed_dim: int = 64):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        # Random Fourier time embedding (fixed).
        self.register_buffer("freqs", torch.randn(time_embed_dim // 2) * 16.0)

        state_dim = input_channels * NUM_STATS
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, time_embed_dim * 2),
            nn.LayerNorm(time_embed_dim * 2),  # stabilises the higher-order moments
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, num_target_channels),
        )

        # Start near unit variance, biasing the HF bands (idx 1..) towards lower variance.
        nn.init.zeros_(self.net[-1].weight)
        prior_bias = torch.zeros(num_target_channels)
        prior_bias[1:] = -2.0
        self.net[-1].bias.data = prior_bias

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Args: ``t`` (B,), ``x`` (B, input_channels, D, H, W).

        Returns ``(log_vars_map, log_vars_flat)`` where the map is broadcast-shaped
        ``(B, num_target_channels, 1, 1, 1)`` for the loss and the flat tensor
        ``(B, num_target_channels)`` is the hint injected into the backbone.
        """
        t = t.view(-1, 1)
        args = t * self.freqs[None]
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        x_flat = x.view(x.shape[0], x.shape[1], -1)
        means = x_flat.mean(dim=2)
        stds = x_flat.std(dim=2)
        maxs = x_flat.abs().max(dim=2)[0]
        l2 = torch.norm(x_flat, p=2, dim=2) / (x_flat.shape[2] ** 0.5)

        safe_std = stds + 1e-6
        centered = x_flat - means.unsqueeze(2)
        skew = (centered**3).mean(dim=2) / (safe_std**3)
        kurt = (centered**4).mean(dim=2) / (safe_std**4)

        state = torch.cat([means, stds, maxs, l2, skew, kurt], dim=1)
        combined = torch.cat([t_emb, self.state_proj(state)], dim=-1)

        raw = self.net(combined)
        log_vars = LOG_VAR_BOUND * torch.tanh(raw / LOG_VAR_BOUND)
        return log_vars.view(t.shape[0], -1, 1, 1, 1), log_vars

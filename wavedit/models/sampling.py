"""ODE integrators for flow-matching sampling.

The flow is integrated from ``t=0`` (noise) to ``t=1`` (data) given a velocity
field ``velocity_fn(t, z) -> dz/dt``. The velocity function owns its own autograd
context (Morpheus guidance differentiates through the uncertainty network), so the
integrators here are plain tensor arithmetic.
"""

from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

VelocityFn = Callable[[float, torch.Tensor], torch.Tensor]


def integrate_ode(velocity_fn: VelocityFn, z0: torch.Tensor, num_steps: int, sampler: str = "heun") -> torch.Tensor:
    """Integrate the probability-flow ODE from ``z0`` over ``num_steps`` uniform steps.

    ``sampler`` is ``"euler"`` (first order) or ``"heun"`` (second-order trapezoidal).
    """
    sampler = sampler.lower()
    if sampler not in ("heun", "euler"):
        raise ValueError(f"Unsupported sampler '{sampler}'. Choose 'heun' or 'euler'.")

    dt = 1.0 / num_steps
    z = z0
    for i in tqdm(range(num_steps), desc=f"WaveDiT {sampler.title()} sampling", leave=False):
        t = i * dt
        d_cur = velocity_fn(t, z)
        if sampler == "euler":
            z = z + dt * d_cur
        else:  # heun: correct with the slope at the tentative Euler endpoint
            z_euler = z + dt * d_cur
            if i < num_steps - 1:
                d_next = velocity_fn(t + dt, z_euler)
                z = z + (dt / 2.0) * (d_cur + d_next)
            else:
                z = z_euler
    return z

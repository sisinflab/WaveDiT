"""Neighborhood attention with a NATTEN-faithful pure-PyTorch fallback.

`NATTEN <https://github.com/SHI-Labs/NATTEN>`_ provides fused CUDA kernels for
neighborhood attention and is the **ground-truth** implementation used whenever it
is installed and running on CUDA. This module adds a dependency-free, device-
agnostic PyTorch fallback (:func:`na2d_torch`) that reproduces NATTEN's ``na2d``
exactly (verified to float32 precision against NATTEN 0.20.0), so WaveDiT also runs
**without** NATTEN — on CPU, or on GPUs lacking a compatible NATTEN build.

NATTEN semantics reproduced here (non-causal, stride 1 — the only case WaveDiT uses):

* Tensor layout ``(B, H, W, heads, head_dim)`` (heads and head_dim last); output
  has the same shape.
* ``logits = scale * (q · k)`` over the neighborhood, softmax over the neighbors,
  then a weighted sum of values. Accumulation is done in float32.
* **Boundary handling is the subtle part:** the ``k``-tap window is *clamped*
  (shifted inward) at the borders, never zero-padded, so every query keeps exactly
  ``k`` neighbors and always attends to itself.

  - dilation 1: ``start = clamp(i - k // 2, 0, L - k)``.
  - dilation ``d > 1``: NATTEN splits each axis into ``d`` residue-class subgrids
    and clamps within each subgrid (``g = i % d``, subgrid length
    ``Lg = ceil((L - g) / d)``, ``start = clamp(i // d - k // 2, 0, Lg - k)``).

  NATTEN raises when ``kernel_size * dilation > L`` on any axis (it never pads);
  the fallback matches that.

Backend is selected by the ``WAVEDIT_NA_BACKEND`` environment variable:
``auto`` (default — NATTEN on CUDA when available, else torch), ``natten`` or ``torch``.
"""

from __future__ import annotations

import os

import torch

try:
    import natten
    import natten.functional as natten_functional

    NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without natten
    natten = None
    natten_functional = None
    NATTEN_AVAILABLE = False

_VALID_BACKENDS = ("auto", "natten", "torch")


def _pair(value) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected a scalar or length-2 value, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def resolve_backend(device: torch.device) -> str:
    """Pick the neighborhood-attention backend for ``device`` (see module docstring)."""
    pref = os.environ.get("WAVEDIT_NA_BACKEND", "auto").lower()
    if pref not in _VALID_BACKENDS:
        raise ValueError(f"WAVEDIT_NA_BACKEND must be one of {_VALID_BACKENDS}, got '{pref}'")
    if pref == "natten":
        if not NATTEN_AVAILABLE:
            raise RuntimeError("WAVEDIT_NA_BACKEND=natten but the natten package is not installed.")
        return "natten"
    if pref == "torch":
        return "torch"
    # auto: NATTEN is the ground-truth base, but it is CUDA-only.
    if NATTEN_AVAILABLE and device.type == "cuda":
        return "natten"
    return "torch"


def _axis_window_starts(length: int, kernel: int, dilation: int, device) -> torch.Tensor:
    """Absolute index of window slot 0 for every query along one axis (NATTEN clamp)."""
    i = torch.arange(length, device=device)
    group = i % dilation
    pos_in_group = torch.div(i, dilation, rounding_mode="floor")
    subgrid_len = torch.div(length - group + dilation - 1, dilation, rounding_mode="floor")
    start_in_group = torch.clamp(pos_in_group - kernel // 2, torch.zeros_like(subgrid_len), subgrid_len - kernel)
    return group + start_in_group * dilation


def na2d_torch(q, k, v, kernel_size, scale=None, dilation=1) -> torch.Tensor:
    """Pure-PyTorch neighborhood attention matching ``natten.functional.na2d``.

    Args mirror NATTEN: ``q, k, v`` are ``(B, H, W, heads, head_dim)``; ``kernel_size``
    and ``dilation`` are int or ``(h, w)``; ``scale`` defaults to ``head_dim ** -0.5``.

    Uses a streaming (flash-style) softmax over the ``k*k`` window slots, so peak
    memory is independent of the kernel size. Accumulates in float32 and returns the
    input dtype.
    """
    if q.ndim != 5:
        raise ValueError(f"expected (B, H, W, heads, head_dim), got shape {tuple(q.shape)}")
    batch, height, width, num_heads, head_dim = q.shape
    kernel_h, kernel_w = _pair(kernel_size)
    dilation_h, dilation_w = _pair(dilation)

    if kernel_h < 2 or kernel_w < 2:
        raise ValueError("kernel_size must be >= 2 along each axis (NATTEN constraint).")
    if kernel_h * dilation_h > height or kernel_w * dilation_w > width:
        raise ValueError(
            "kernel_size * dilation must be <= input size along each axis "
            f"(kernel={kernel_h, kernel_w}, dilation={dilation_h, dilation_w}, grid={height, width})."
        )
    scale = float(head_dim ** -0.5 if scale is None else scale)

    out_dtype = q.dtype
    query = q.float() * scale
    key = k.float()
    value = v.float()

    start_h = _axis_window_starts(height, kernel_h, dilation_h, q.device)
    start_w = _axis_window_starts(width, kernel_w, dilation_w, q.device)

    # Streaming softmax accumulators (all float32).
    running_max = torch.full((batch, height, width, num_heads), float("-inf"), device=q.device)
    running_sum = torch.zeros((batch, height, width, num_heads), device=q.device)
    accumulator = torch.zeros((batch, height, width, num_heads, head_dim), device=q.device)

    for slot_h in range(kernel_h):
        rows = start_h + slot_h * dilation_h
        key_rows = key.index_select(1, rows)
        value_rows = value.index_select(1, rows)
        for slot_w in range(kernel_w):
            cols = start_w + slot_w * dilation_w
            key_window = key_rows.index_select(2, cols)
            value_window = value_rows.index_select(2, cols)

            logits = (query * key_window).sum(-1)  # (B, H, W, heads)
            new_max = torch.maximum(running_max, logits)
            decay = torch.exp(running_max - new_max)
            weight = torch.exp(logits - new_max)
            running_sum = running_sum * decay + weight
            accumulator = accumulator * decay.unsqueeze(-1) + weight.unsqueeze(-1) * value_window
            running_max = new_max

    output = accumulator / running_sum.unsqueeze(-1)
    return output.to(out_dtype)


def na2d_natten(q, k, v, kernel_size, scale=None, dilation=1) -> torch.Tensor:
    """The ground-truth NATTEN ``na2d`` (requires natten + a CUDA tensor)."""
    if not NATTEN_AVAILABLE:
        raise RuntimeError("natten is not installed.")
    return natten_functional.na2d(q, k, v, kernel_size, dilation=dilation, scale=scale)


def na2d(q, k, v, kernel_size, scale=None, dilation=1) -> torch.Tensor:
    """Neighborhood attention dispatching to NATTEN (ground truth) or the torch fallback."""
    if resolve_backend(q.device) == "natten":
        return na2d_natten(q, k, v, kernel_size, scale=scale, dilation=dilation)
    return na2d_torch(q, k, v, kernel_size, scale=scale, dilation=dilation)

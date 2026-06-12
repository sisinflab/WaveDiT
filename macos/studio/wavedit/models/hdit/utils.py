"""Minimal tensor helpers for the HDiT backbone.

This is a trimmed-down version of k-diffusion's ``utils`` module
(https://github.com/crowsonkb/k-diffusion). Only ``append_dims`` is used by the
backbone; the original module's EMA/LR-schedule/dataset/checkpoint utilities have
been removed to keep the published codebase lean and dependency-light.
"""

from __future__ import annotations


def append_dims(x, target_dims):
    """Append trailing singleton dims to ``x`` until it has ``target_dims`` dims."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

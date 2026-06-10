"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs.

    ``torch`` is imported lazily so that lightweight, torch-free consumers of this
    package (e.g. config parsing/tests) do not pull in the heavy dependency.

    We deliberately leave ``cudnn.deterministic`` off: full determinism noticeably
    slows down the 3D attention kernels and is not required for the experiments.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

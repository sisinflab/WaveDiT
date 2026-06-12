"""Differentiable separable 3D discrete wavelet transform (DWT / IDWT).

Adapted from https://github.com/pfriedri/wdm-3d/tree/main/DWT_IDWT (thanks to the
authors). The forward/backward matrix algebra is kept numerically identical to the
original; only naming, caching and formatting have been modernised.

A single ``haar`` level maps a volume ``(N, C, D, H, W)`` to its eight octant
sub-bands ``(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)`` each of shape
``(N, C, D/2, H/2, W/2)``. WaveDiT concatenates the eight sub-bands along the
channel axis to obtain an 8-channel latent.
"""

from __future__ import annotations

import math

import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.autograd import Function


class _DWTFunction3D(Function):
    @staticmethod
    def forward(ctx, x, low_0, low_1, low_2, high_0, high_1, high_2):
        ctx.save_for_backward(low_0, low_1, low_2, high_0, high_1, high_2)
        L = torch.matmul(low_0, x)
        H = torch.matmul(high_0, x)
        LL = torch.matmul(L, low_1).transpose(2, 3)
        LH = torch.matmul(L, high_1).transpose(2, 3)
        HL = torch.matmul(H, low_1).transpose(2, 3)
        HH = torch.matmul(H, high_1).transpose(2, 3)
        LLL = torch.matmul(low_2, LL).transpose(2, 3)
        LLH = torch.matmul(low_2, LH).transpose(2, 3)
        LHL = torch.matmul(low_2, HL).transpose(2, 3)
        LHH = torch.matmul(low_2, HH).transpose(2, 3)
        HLL = torch.matmul(high_2, LL).transpose(2, 3)
        HLH = torch.matmul(high_2, LH).transpose(2, 3)
        HHL = torch.matmul(high_2, HL).transpose(2, 3)
        HHH = torch.matmul(high_2, HH).transpose(2, 3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, g_LLL, g_LLH, g_LHL, g_LHH, g_HLL, g_HLH, g_HHL, g_HHH):
        low_0, low_1, low_2, high_0, high_1, high_2 = ctx.saved_tensors
        dtype = g_LLL.dtype
        l0, l1, l2 = low_0.to(dtype), low_1.to(dtype), low_2.to(dtype)
        h0, h1, h2 = high_0.to(dtype), high_1.to(dtype), high_2.to(dtype)

        g_LL = torch.add(torch.matmul(l2.t(), g_LLL.transpose(2, 3)), torch.matmul(h2.t(), g_HLL.transpose(2, 3))).transpose(2, 3)
        g_LH = torch.add(torch.matmul(l2.t(), g_LLH.transpose(2, 3)), torch.matmul(h2.t(), g_HLH.transpose(2, 3))).transpose(2, 3)
        g_HL = torch.add(torch.matmul(l2.t(), g_LHL.transpose(2, 3)), torch.matmul(h2.t(), g_HHL.transpose(2, 3))).transpose(2, 3)
        g_HH = torch.add(torch.matmul(l2.t(), g_LHH.transpose(2, 3)), torch.matmul(h2.t(), g_HHH.transpose(2, 3))).transpose(2, 3)
        g_L = torch.add(torch.matmul(g_LL, l1.t()), torch.matmul(g_LH, h1.t()))
        g_H = torch.add(torch.matmul(g_HL, l1.t()), torch.matmul(g_HH, h1.t()))
        g_x = torch.add(torch.matmul(l0.t(), g_L), torch.matmul(h0.t(), g_H))
        return g_x, None, None, None, None, None, None


class _IDWTFunction3D(Function):
    @staticmethod
    def forward(ctx, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, low_0, low_1, low_2, high_0, high_1, high_2):
        ctx.save_for_backward(low_0, low_1, low_2, high_0, high_1, high_2)
        LL = torch.add(torch.matmul(low_2.t(), LLL.transpose(2, 3)), torch.matmul(high_2.t(), HLL.transpose(2, 3))).transpose(2, 3)
        LH = torch.add(torch.matmul(low_2.t(), LLH.transpose(2, 3)), torch.matmul(high_2.t(), HLH.transpose(2, 3))).transpose(2, 3)
        HL = torch.add(torch.matmul(low_2.t(), LHL.transpose(2, 3)), torch.matmul(high_2.t(), HHL.transpose(2, 3))).transpose(2, 3)
        HH = torch.add(torch.matmul(low_2.t(), LHH.transpose(2, 3)), torch.matmul(high_2.t(), HHH.transpose(2, 3))).transpose(2, 3)
        L = torch.add(torch.matmul(LL, low_1.t()), torch.matmul(LH, high_1.t()))
        H = torch.add(torch.matmul(HL, low_1.t()), torch.matmul(HH, high_1.t()))
        return torch.add(torch.matmul(low_0.t(), L), torch.matmul(high_0.t(), H))

    @staticmethod
    def backward(ctx, grad_output):
        low_0, low_1, low_2, high_0, high_1, high_2 = ctx.saved_tensors
        dtype = grad_output.dtype
        l0, l1, l2 = low_0.to(dtype), low_1.to(dtype), low_2.to(dtype)
        h0, h1, h2 = high_0.to(dtype), high_1.to(dtype), high_2.to(dtype)

        g_L = torch.matmul(l0, grad_output)
        g_H = torch.matmul(h0, grad_output)
        g_LL = torch.matmul(g_L, l1).transpose(2, 3)
        g_LH = torch.matmul(g_L, h1).transpose(2, 3)
        g_HL = torch.matmul(g_H, l1).transpose(2, 3)
        g_HH = torch.matmul(g_H, h1).transpose(2, 3)
        g_LLL = torch.matmul(l2, g_LL).transpose(2, 3)
        g_LLH = torch.matmul(l2, g_LH).transpose(2, 3)
        g_LHL = torch.matmul(l2, g_HL).transpose(2, 3)
        g_LHH = torch.matmul(l2, g_HH).transpose(2, 3)
        g_HLL = torch.matmul(h2, g_LL).transpose(2, 3)
        g_HLH = torch.matmul(h2, g_LH).transpose(2, 3)
        g_HHL = torch.matmul(h2, g_HL).transpose(2, 3)
        g_HHH = torch.matmul(h2, g_HH).transpose(2, 3)
        return g_LLL, g_LLH, g_LHL, g_LHH, g_HLL, g_HLH, g_HHL, g_HHH, None, None, None, None, None, None


def _build_band_matrices(shape, band_low, band_high, band_length, band_length_half, device):
    """Build the per-axis low/high analysis matrices for a given spatial ``shape``."""
    depth, height, width = shape[-3:]
    longest = max(depth, height, width)
    half = math.floor(longest / 2)
    cols = longest + band_length - 2
    end = None if band_length_half == 1 else (-band_length_half + 1)

    matrix_low = np.zeros((half, cols))
    index = 0
    for i in range(half):
        for j in range(band_length):
            matrix_low[i, index + j] = band_low[j]
        index += 2

    matrix_high = np.zeros((longest - half, cols))
    index = 0
    for i in range(longest - half):
        for j in range(band_length):
            matrix_high[i, index + j] = band_high[j]
        index += 2

    low_h = matrix_low[0:math.floor(height / 2), 0:(height + band_length - 2)]
    low_w = matrix_low[0:math.floor(width / 2), 0:(width + band_length - 2)]
    low_d = matrix_low[0:math.floor(depth / 2), 0:(depth + band_length - 2)]
    high_h = matrix_high[0:(height - math.floor(height / 2)), 0:(height + band_length - 2)]
    high_w = matrix_high[0:(width - math.floor(width / 2)), 0:(width + band_length - 2)]
    high_d = matrix_high[0:(depth - math.floor(depth / 2)), 0:(depth + band_length - 2)]

    crop = slice(band_length_half - 1, end)
    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return {
        "low_0": to_tensor(low_h[:, crop]),
        "low_1": to_tensor(np.transpose(low_w[:, crop])),
        "low_2": to_tensor(low_d[:, crop]),
        "high_0": to_tensor(high_h[:, crop]),
        "high_1": to_tensor(np.transpose(high_w[:, crop])),
        "high_2": to_tensor(high_d[:, crop]),
    }


class DWT3D(nn.Module):
    """Forward 3D discrete wavelet transform."""

    def __init__(self, wavename: str = "haar"):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self._cache: dict = {}

    def _matrices(self, shape, device):
        # Key on (shape, device): the analysis matrices depend on the spatial shape,
        # so a single shared "last shape" would return wrong-sized matrices when more
        # than one shape (or device) is used.
        key = (tuple(shape), str(device))
        if key not in self._cache:
            self._cache[key] = _build_band_matrices(
                shape, self.band_low, self.band_high, self.band_length, self.band_length_half, device
            )
        return self._cache[key]

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5, "DWT3D expects (N, C, D, H, W)"
        if x.dtype != torch.float32:
            x = x.float()
        m = self._matrices(tuple(x.shape), x.device)
        return _DWTFunction3D.apply(x, m["low_0"], m["low_1"], m["low_2"], m["high_0"], m["high_1"], m["high_2"])


class IDWT3D(nn.Module):
    """Inverse 3D discrete wavelet transform."""

    def __init__(self, wavename: str = "haar"):
        super().__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = list(wavelet.dec_lo)
        self.band_high = list(wavelet.dec_hi)
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self._cache: dict = {}

    def _matrices(self, shape, device):
        # Key on (shape, device): the analysis matrices depend on the spatial shape,
        # so a single shared "last shape" would return wrong-sized matrices when more
        # than one shape (or device) is used.
        key = (tuple(shape), str(device))
        if key not in self._cache:
            self._cache[key] = _build_band_matrices(
                shape, self.band_low, self.band_high, self.band_length, self.band_length_half, device
            )
        return self._cache[key]

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert LLL.dim() == 5, "IDWT3D expects (N, C, D/2, H/2, W/2)"
        target_shape = (
            LLL.size(0),
            LLL.size(1),
            LLL.size(-3) + HHH.size(-3),
            LLL.size(-2) + HHH.size(-2),
            LLL.size(-1) + HHH.size(-1),
        )
        m = self._matrices(target_shape, LLL.device)
        return _IDWTFunction3D.apply(
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
            m["low_0"], m["low_1"], m["low_2"], m["high_0"], m["high_1"], m["high_2"],
        )


# Module-level singletons used throughout the codebase.
dwt_3d = DWT3D(wavename="haar")
idwt_3d = IDWT3D(wavename="haar")

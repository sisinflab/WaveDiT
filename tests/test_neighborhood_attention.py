"""Verify the pure-PyTorch neighborhood-attention fallback against NATTEN.

    python tests/test_neighborhood_attention.py

NATTEN (CUDA-only) is the ground truth. On a CUDA machine with NATTEN installed,
this exhaustively checks numerical equivalence across shapes/kernels/dilations/dtypes.
The CPU fallback, dispatcher, gradients and an end-to-end no-NATTEN model forward are
checked everywhere.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("K_DIFFUSION_USE_COMPILE", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from wavedit.models.hdit.neighborhood_attention import (
    NATTEN_AVAILABLE,
    na2d_natten,
    na2d_torch,
    resolve_backend,
)


def _rand(b, h, w, nh, d, device, dtype):
    g = torch.Generator(device=device).manual_seed(1234 + h * 31 + w * 17 + nh + d)
    return [torch.randn(b, h, w, nh, d, device=device, generator=g, dtype=dtype) for _ in range(3)]


def test_equivalence_vs_natten_cuda():
    if not (torch.cuda.is_available() and NATTEN_AVAILABLE):
        print("  SKIP equivalence (needs CUDA + natten)")
        return
    dev = torch.device("cuda")
    sizes = [7, 8, 14, 16]
    kernels = [3, 5, 7]
    worst = 0.0
    n_cases = 0
    for h, w, k, nh, d, b in itertools.product(sizes, sizes, kernels, [1, 4], [16, 64], [1, 2]):
        if k > h or k > w:
            continue
        q, kk, v = _rand(b, h, w, nh, d, dev, torch.float32)
        ref = na2d_natten(q, kk, v, k, scale=1.0)
        got = na2d_torch(q, kk, v, k, scale=1.0)
        diff = (ref - got).abs().max().item()
        worst = max(worst, diff)
        n_cases += 1
        assert diff < 1e-3, f"fp32 mismatch {diff} at h{h} w{w} k{k} nh{nh} d{d} b{b}"
    print(f"  OK fp32 equivalence: {n_cases} cases, worst |diff|={worst:.2e}")

    # Dilation > 1 (residue-class clamping)
    worst_d = 0.0
    n_d = 0
    for h, k, dil in itertools.product([14, 16], [3, 5], [2, 3]):
        if k * dil > h:
            continue
        q, kk, v = _rand(2, h, h, 4, 64, dev, torch.float32)
        ref = na2d_natten(q, kk, v, k, scale=1.0, dilation=dil)
        got = na2d_torch(q, kk, v, k, scale=1.0, dilation=dil)
        d_ = (ref - got).abs().max().item()
        worst_d = max(worst_d, d_)
        n_d += 1
        assert d_ < 1e-3, f"dilation mismatch {d_} at h{h} k{k} dil{dil}"
    print(f"  OK dilation>1 equivalence: {n_d} cases, worst |diff|={worst_d:.2e}")

    # bf16 (model's training dtype): looser tolerance, both accumulate in fp32 internally.
    worst_bf = 0.0
    for h, k in itertools.product([8, 14], [3, 7]):
        q, kk, v = _rand(2, h, h, 4, 64, dev, torch.bfloat16)
        ref = na2d_natten(q, kk, v, k, scale=1.0).float()
        got = na2d_torch(q, kk, v, k, scale=1.0).float()
        worst_bf = max(worst_bf, (ref - got).abs().max().item())
    assert worst_bf < 5e-2, f"bf16 mismatch too large: {worst_bf}"
    print(f"  OK bf16 equivalence: worst |diff|={worst_bf:.2e}")


def test_boundary_stress_cuda():
    """Small grids where edge clamping dominates (window == axis, axis = k+1)."""
    if not (torch.cuda.is_available() and NATTEN_AVAILABLE):
        print("  SKIP boundary stress (needs CUDA + natten)")
        return
    dev = torch.device("cuda")
    for k in [3, 5, 7]:
        for h in [k, k + 1, k + 2]:
            q, kk, v = _rand(1, h, h, 2, 32, dev, torch.float32)
            ref = na2d_natten(q, kk, v, k, scale=1.0)
            got = na2d_torch(q, kk, v, k, scale=1.0)
            assert (ref - got).abs().max().item() < 1e-3, f"boundary mismatch k{k} h{h}"
    print("  OK boundary stress (window==axis and near-edge)")


def test_cpu_matches_gpu_fallback():
    """The torch fallback must be device-agnostic: CPU result == CUDA fallback result."""
    q, k, v = _rand(2, 14, 14, 4, 64, torch.device("cpu"), torch.float32)
    out_cpu = na2d_torch(q, k, v, 7, scale=1.0)
    assert torch.isfinite(out_cpu).all() and out_cpu.shape == q.shape
    if torch.cuda.is_available():
        out_gpu = na2d_torch(q.cuda(), k.cuda(), v.cuda(), 7, scale=1.0).cpu()
        diff = (out_cpu - out_gpu).abs().max().item()
        assert diff < 1e-4, f"CPU vs CUDA fallback diff {diff}"
        print(f"  OK CPU fallback runs and matches CUDA fallback (|diff|={diff:.2e})")
    else:
        print("  OK CPU fallback runs (no CUDA to cross-check)")


def test_dispatcher_and_validation():
    cpu = torch.device("cpu")
    assert resolve_backend(cpu) == "torch"  # natten is CUDA-only
    prev = os.environ.get("WAVEDIT_NA_BACKEND")
    try:
        os.environ["WAVEDIT_NA_BACKEND"] = "torch"
        assert resolve_backend(torch.device("cuda")) == "torch"
        if NATTEN_AVAILABLE and torch.cuda.is_available():
            os.environ["WAVEDIT_NA_BACKEND"] = "natten"
            assert resolve_backend(torch.device("cuda")) == "natten"
    finally:
        if prev is None:
            os.environ.pop("WAVEDIT_NA_BACKEND", None)
        else:
            os.environ["WAVEDIT_NA_BACKEND"] = prev
    # NATTEN's constraint: kernel*dilation must fit; fallback must raise too.
    q = torch.randn(1, 5, 5, 1, 16)
    for bad in (lambda: na2d_torch(q, q, q, 7), lambda: na2d_torch(q, q, q, 3, dilation=2)):
        try:
            bad()
            raise AssertionError("expected ValueError for kernel*dilation > size")
        except ValueError:
            pass
    print("  OK dispatcher selection + input validation")


def test_gradients():
    q, k, v = _rand(1, 8, 8, 2, 16, torch.device("cpu"), torch.float32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    na2d_torch(q, k, v, 5, scale=1.0).sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    print("  OK fallback is differentiable (finite grads)")


def test_end_to_end_model_without_natten():
    """Full WaveDiT forward/backward + sampling using the torch backend (no NATTEN)."""
    from wavedit import Config
    from wavedit.models import build_model

    prev = os.environ.get("WAVEDIT_NA_BACKEND")
    os.environ["WAVEDIT_NA_BACKEND"] = "torch"  # force the fallback even if natten is present
    try:
        cfg = Config.from_dict({
            "device": "cpu",
            "data": {"data_folder": "x", "conditions": {"age": "numeric"}, "image_size": [128, 128, 128]},
            "model": {
                "patch_size": [8, 8], "cond_embed_dim": 32, "slice_embed_dim": 32, "flow": "cfm",
                "levels": [
                    {"depth": 1, "width": 64, "d_ff": 128, "self_attn": {"type": "neighborhood", "d_head": 32, "kernel_size": 7}},
                    {"depth": 1, "width": 64, "d_ff": 128, "self_attn": {"type": "spatio-temporal", "d_head": 32}},
                ],
                "mapping": {"depth": 1, "width": 64, "d_ff": 128},
            },
            "train": {"num_flow_steps_sampling": 2},
        })
        model = build_model(cfg, {"age": {"type": "numeric"}}, {"age": {"min": 5.0, "max": 95.0}}, {}, {"age": 50.0})
        images = torch.randn(1, 1, 128, 128, 128)
        loss = model.loss(images, {"age": torch.tensor([[0.5]])}, 0)
        loss.backward()
        assert torch.isfinite(loss), loss
        model.eval()
        sample = model.sample(num_samples=1, raw_conditions={"age": 60.0}, cfg_scale=1.0, morpheus_scale=0.0)
        assert sample.shape == (1, 1, 128, 128, 128) and torch.isfinite(sample).all()
        print(f"  OK end-to-end CPU model with neighborhood attention via torch fallback (loss={loss.item():.3f})")
    finally:
        if prev is None:
            os.environ.pop("WAVEDIT_NA_BACKEND", None)
        else:
            os.environ["WAVEDIT_NA_BACKEND"] = prev


if __name__ == "__main__":
    print("natten available:", NATTEN_AVAILABLE, "| cuda:", torch.cuda.is_available())
    test_equivalence_vs_natten_cuda()
    test_boundary_stress_cuda()
    test_cpu_matches_gpu_fallback()
    test_dispatcher_and_validation()
    test_gradients()
    test_end_to_end_model_without_natten()
    print("\nNEIGHBORHOOD ATTENTION TESTS PASSED ✔")

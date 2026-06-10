"""Pre-compute the hero brain shown in the viewer on first page load.

Run this ONCE on a GPU machine (where torch + the wavedit deps are installed), then
commit the resulting assets/wavedit_hero_age45_base_seed42.nii.gz to the Space repo.
On first paint the app injects this file so the viewer is never empty; the first
Generate replaces it.

    python scripts/make_hero.py

The output is a small uint8 gzipped NIfTI (a few MB) - the same format the live app
feeds the viewer - so it loads instantly.
"""

from __future__ import annotations

import os

os.environ.setdefault("WAVEDIT_NA_BACKEND", "torch")

import gzip
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from wavedit import Config
from wavedit.models import build_model
from wavedit.training.checkpoint import load_model_weights
from wavedit.generation.generator import center_crop_bounds

HF_REPO = "danesed/WaveDiT"
HF_REVISION = "main"
AGE, SEED, STEPS = 45.0, 42, 10
OUT = Path(__file__).resolve().parent.parent / "assets" / "wavedit_hero_age45_base_seed42.nii.gz"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = hf_hub_download(HF_REPO, "WaveDiT-Base.pth", revision=HF_REVISION)
    ck = torch.load(path, map_location="cpu", weights_only=True)

    cfg = Config.from_dict(ck["config"])
    model = build_model(
        cfg, ck["condition_config"], ck["condition_ranges"],
        ck["categorical_maps"], ck["null_conditions"], num_flow_steps=STEPS,
    )
    load_model_weights(model, ck)
    model.to(device).eval()

    torch.manual_seed(SEED)
    with torch.no_grad():
        vol = model.sample(
            num_samples=1, raw_conditions={"age": AGE}, cfg_scale=1.0,
            sampler="heun", morpheus_scale=None, cfg_rescale=0.7,
            autocast_dtype=torch.bfloat16,
        )

    vol = torch.clamp((vol.float() + 1.0) / 2.0, 0.0, 1.0)
    full = tuple(int(s) for s in cfg.data.image_size)
    (d0, d1), (h0, h1), (w0, w1) = center_crop_bounds(full, (182, 218, 182))
    arr = vol[:, :, d0:d1, h0:h1, w0:w1][0, 0].cpu().numpy().astype(np.float32)

    u8 = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
    raw = nib.Nifti1Image(u8, np.eye(4)).to_bytes()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(gzip.compress(raw, compresslevel=6))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

"""End-to-end CPU smoke test for WaveDiT.

Synthesises a tiny NIfTI dataset, then runs the full pipeline (data -> model -> one
training epoch -> checkpoint -> reload -> sampling -> save). It uses small,
natten-free attention so it runs anywhere PyTorch is installed.

    python tests/smoke_test.py

Requires: torch, nibabel, monai, pywavelets, einops, scikit-image. It does NOT
require a GPU, natten or wandb. This validates wiring/shapes, not model quality.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Disable torch.compile for a fast, portable CPU run (must precede model imports).
os.environ.setdefault("K_DIFFUSION_USE_COMPILE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from wavedit import Config
from wavedit.data import build_datasets, collate_fn
from wavedit.generation import generate_samples, parse_condition_sets
from wavedit.models import build_model
from wavedit.training import Trainer, load_model_weights

IMAGE_SIZE = 64  # -> 32^3 wavelet latent

SMOKE_CONFIG = {
    "run_name": "smoke",
    "seed": 0,
    "device": "cpu",
    "precision": "fp32",
    "data": {
        "data_folder": None,  # set at runtime to the synthetic folder
        "conditions": {"age": "numeric"},
        "image_size": [IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE],
        "val_split": 0.5,
        "num_workers": 0,
    },
    "model": {
        "patch_size": [8, 8],
        "cond_embed_dim": 32,
        "slice_embed_dim": 32,
        "flow": "cfm",
        "morpheus_scale": 1.0,
        "levels": [
            {"depth": 1, "width": 64, "d_ff": 128, "dropout": 0.0, "self_attn": {"type": "global", "d_head": 16}},
            {"depth": 1, "width": 64, "d_ff": 128, "dropout": 0.0, "self_attn": {"type": "spatio-temporal", "d_head": 16}},
        ],
        "mapping": {"depth": 1, "width": 64, "d_ff": 128, "dropout": 0.0},
    },
    "train": {"epochs": 1, "batch_size": 1, "num_flow_steps_sampling": 2, "early_stop_patience": 0},
    "sampling": {"sampler": "heun", "cfg_rescale": 0.7},
    "logging": {"wandb": False},
    "post_train_generation": {"enabled": False},
}


def synthesize_dataset(folder: Path, num_subjects: int = 4) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, age in enumerate(np.linspace(20, 80, num_subjects)):
        volume = rng.standard_normal((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), folder / f"subj{i:02d}_AGE_{age:.1f}.nii.gz")


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "scans"
        synthesize_dataset(data_dir)

        cfg_dict = {**SMOKE_CONFIG}
        cfg_dict["data"] = {**SMOKE_CONFIG["data"], "data_folder": str(data_dir)}
        cfg_dict["logging"] = {**SMOKE_CONFIG["logging"], "checkpoint_dir": str(tmp / "checkpoints")}
        cfg = Config.from_dict(cfg_dict)

        print("[1/6] Building datasets ...")
        bundle = build_datasets(cfg.data, cfg.seed)
        train_loader = DataLoader(bundle.train, batch_size=1, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(bundle.val, batch_size=1, collate_fn=collate_fn)
        images, conditions = next(iter(train_loader))
        assert images.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), images.shape
        assert conditions["age"].shape == (1, 1), conditions["age"].shape
        print("      batch image", tuple(images.shape), "| age", conditions["age"].flatten().tolist())

        print("[2/6] Building model ...")
        model = build_model(cfg, bundle.condition_config, bundle.condition_ranges,
                            bundle.categorical_maps, bundle.null_conditions)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      parameters: {n_params:,}")

        print("[3/6] One forward/backward step ...")
        loss = model.loss(images, conditions, global_step=0)
        loss.backward()
        assert torch.isfinite(loss), loss
        print(f"      loss = {loss.item():.4f}")

        print("[4/6] One training epoch via Trainer (+ checkpointing) ...")
        run_dir = Path(cfg.logging.checkpoint_dir) / cfg.run_name
        metadata = {
            "config": cfg.to_dict(),
            "condition_config": bundle.condition_config,
            "condition_ranges": bundle.condition_ranges,
            "categorical_maps": bundle.categorical_maps,
            "cardinalities": bundle.cardinalities,
            "null_conditions": bundle.null_conditions,
        }
        Trainer(model, train_loader, val_loader, cfg, run_dir, metadata).fit()
        assert (run_dir / "best.pth").exists() and (run_dir / "last.pth").exists()
        print("      checkpoints written:", os.listdir(run_dir))

        print("[5/6] Reload checkpoint into a fresh model and sample ...")
        checkpoint = torch.load(run_dir / "best.pth", map_location="cpu")
        reloaded = build_model(Config.from_dict(checkpoint["config"]),
                               checkpoint["condition_config"], checkpoint["condition_ranges"],
                               checkpoint["categorical_maps"], checkpoint["null_conditions"])
        load_model_weights(reloaded, checkpoint)
        reloaded.eval()
        sample = reloaded.sample(num_samples=1, raw_conditions={"age": 50.0}, cfg_scale=1.5)
        assert sample.shape == (1, 1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), sample.shape
        assert torch.isfinite(sample).all()
        print("      sample", tuple(sample.shape), "range", (sample.min().item(), sample.max().item()))

        print("[6/6] generate_samples writes NIfTI files ...")
        out_dir = tmp / "generated"
        conds = parse_condition_sets(["age=30.0", "age=70.0"], reloaded.condition_config)
        generate_samples(reloaded, conds, num_samples_per_condition=1, output_dir=out_dir,
                         save_size=(48, 48, 48), model_output_size=cfg.data.image_size, cfg_scale=1.0)
        written = list(out_dir.rglob("*.nii.gz"))
        assert len(written) == 2, written
        print("      wrote", [p.name for p in written])

    print("\nSMOKE TEST PASSED ✔")


if __name__ == "__main__":
    main()

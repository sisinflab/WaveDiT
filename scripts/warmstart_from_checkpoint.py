#!/usr/bin/env python3
"""Warm-start a WaveDiT model from a pretrained checkpoint via shape-matched weight transfer.

#TLDR; Warm-start a WaveDiT model from a pretrained donor that differs only in
# patch size, saving hours of training. Works both ways: 
# small->big (coarse donor -> finer target) and big->small (fine donor -> coarser target).

================================================================================
WHY THIS WORKS
================================================================================
WaveDiT's transformer body (neighborhood / spatio-temporal attention, the
feed-forward blocks, the mapping network and the conditioning embedders) is
*resolution agnostic*: token positions are produced on the fly by
``make_axial_pos`` and the rotary frequencies depend only on the attention head
dimension, never on the token-grid size. The only parameters tied to the patch
size are the two patch projections, ``patch_in`` and ``patch_out``.

Consequently a model trained at one patch size can hand almost all of its weights
to a model with a different patch size, in either direction, coarse->fine
(e.g. 8x8 -> 2x2) or fine->coarse (e.g. 2x2 -> 8x8). Only the two patch
projections must be re-initialised and re-learned; the expensive body starts
already trained instead of from random noise. In practice this transfers
~95 / 97 tensors and converges dramatically faster than training from scratch.

The same mechanism doubles as a general "weight surgery" tool: any donor that
shares part of the target's architecture contributes every tensor whose name and
shape match, and the rest keep their fresh initialisation.

================================================================================
WHAT IT DOES
================================================================================
1. Builds the *target* model purely from its YAML config. The dataset-derived
   condition metadata is read from the donor checkpoint, so no dataset access is
   required.
2. Copies every donor tensor whose name AND shape match the target. Every other
   target tensor keeps its fresh initialisation. All three sets (transferred /
   reinitialised / donor-only) are reported.
3. Writes a self-contained checkpoint with ``epoch = -1`` so the standard
   training loop resumes it at epoch 0 with a *fresh optimizer*. Point the target
   config's ``train.resume_from`` at the output file and train as usual.

================================================================================
USAGE
================================================================================
    python scripts/warmstart_from_checkpoint.py \
        --donor  checkpoints/WaveDiT/WaveDiT-FinePatch.pth \
        --config configs/cfm_Patch2.yaml \
        --output checkpoints/WaveDiT_CFM_Patch2/last.pth

Then make sure the target config contains:

    train:
      resume_from: ./checkpoints/WaveDiT_CFM_Patch2/last.pth

and launch training normally. The training log should report
``Resumed at epoch 0``.

================================================================================
CAVEATS
================================================================================
* Donor and target should share the transformer-body shape (depth, width, d_ff,
  attention types) for a full-body transfer. A donor with a different depth or
  width still works but contributes only the tensors that match exactly.
* Changing the patch size changes the physical receptive field of neighborhood
  attention and the token-grid size (either way), so a short adaptation phase is
  expected. It is still far ahead of random initialisation.
* Only model weights are transferred; the optimizer restarts from scratch, which
  is the recommended behaviour when the patch size (and thus part of the
  parameter set) changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/warmstart_from_checkpoint.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wavedit import Config
from wavedit.models import build_model
from wavedit.training.checkpoint import strip_wrapper_prefixes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warm-start a WaveDiT model from a pretrained checkpoint "
                    "via shape-matched weight transfer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--donor", required=True,
                   help="Path to the pretrained donor checkpoint (.pth).")
    p.add_argument("--config", required=True,
                   help="YAML config of the TARGET model to warm-start.")
    p.add_argument("--output", required=True,
                   help="Where to write the warm-start checkpoint "
                        "(typically the target run's last.pth).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    donor = torch.load(args.donor, map_location="cpu")
    if "model_state_dict" not in donor:
        raise KeyError(f"Donor checkpoint {args.donor!r} has no 'model_state_dict'.")
    for key in ("condition_config", "condition_ranges", "categorical_maps", "null_conditions"):
        if key not in donor:
            raise KeyError(f"Donor checkpoint is missing '{key}'; cannot rebuild the model.")

    cfg = Config.from_yaml(args.config)

    # The target model is defined entirely by its config; condition metadata comes
    # from the donor (donor and target are assumed to share the same conditioning).
    model = build_model(
        cfg,
        donor["condition_config"],
        donor["condition_ranges"],
        donor["categorical_maps"],
        donor["null_conditions"],
    )

    src = strip_wrapper_prefixes(donor["model_state_dict"])
    tgt = model.state_dict()

    # Shape-matched transfer: keep every donor tensor whose name and shape match.
    transfer, reinit_shape, reinit_absent = {}, [], []
    for name, tensor in tgt.items():
        if name in src and src[name].shape == tensor.shape:
            transfer[name] = src[name]
        elif name in src:
            reinit_shape.append(f"{name}  donor{tuple(src[name].shape)} != target{tuple(tensor.shape)}")
        else:
            reinit_absent.append(name)
    donor_only = [name for name in src if name not in tgt]

    missing, unexpected = model.load_state_dict(transfer, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"

    print(f"Donor : {args.donor}")
    print(f"Target: {args.config}")
    print(f"Transferred (name+shape match): {len(transfer)} / {len(tgt)} tensors")
    if reinit_shape:
        print(f"Re-initialised (shape mismatch): {len(reinit_shape)}")
        for line in reinit_shape:
            print(f"    - {line}")
    if reinit_absent:
        print(f"Re-initialised (absent in donor): {len(reinit_absent)}")
        for name in reinit_absent:
            print(f"    - {name}")
    if donor_only:
        print(f"Ignored (donor-only, not in target): {len(donor_only)}")
        for name in donor_only:
            print(f"    - {name}")

    payload = {
        "epoch": -1,  # -> training resumes at epoch 0 with a fresh optimizer
        "model_state_dict": model.state_dict(),
        "best_val_loss": float("inf"),
        "epochs_without_improvement": 0,
        "config": cfg.to_dict(),
        "condition_config": donor["condition_config"],
        "condition_ranges": donor["condition_ranges"],
        "categorical_maps": donor["categorical_maps"],
        "cardinalities": donor.get("cardinalities", {}),
        "null_conditions": donor["null_conditions"],
        "note": f"warm-start from {Path(args.donor).name} via shape-matched weight transfer",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Saved warm-start checkpoint -> {out}")
    print(f"Trainable params: {n:,}")
    print("Set the target config's train.resume_from to this file and start training "
          "(the log should report 'Resumed at epoch 0').")


if __name__ == "__main__":
    main()

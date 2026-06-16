#!/usr/bin/env python3
"""Warm-start a WaveDiT model from a pretrained checkpoint by weight inheritance.

#TLDR; Hand almost all of a trained WaveDiT's weights to another WaveDiT that
# differs only in patch size. The transformer body is copied 1:1 and the two patch
# projections are resized to the new patch with the FlexiViT PI-resize, so even the
# tokenizer/detokenizer start trained instead of from scratch, saving hours.
#
# Use this to warm-start a bigger model with a finer patch from a smaller, coarser one.
# Or even to distill a well-trained fine-patch model into a cheap coarse-patch one by init.

================================================================================
WHY THIS WORKS
================================================================================
WaveDiT's transformer body (neighborhood / spatio-temporal attention, the
feed-forward blocks, the mapping network and the conditioning embedders) is
*resolution agnostic*: token positions are produced on the fly by
``make_axial_pos`` and the rotary frequencies depend only on the attention head
dimension, never on the token-grid size. The only parameters tied to the patch
size are the two patch projections, ``patch_in`` and ``patch_out``.

Those two are just spatial p x p linear filters, so a model trained at one patch
size can hand *all* of its weights to a model with a different patch size: the
body transfers by exact name+shape match, and the two patch projections are
remapped with the **FlexiViT PI-resize** (Beyer et al., 2023), the
pseudo-inverse of the bilinear patch-resize operator, the least-squares-optimal
way to change a patch-embedding's resolution. Because a token is an inner product
<w, patch>, naively resizing the weight does not preserve the function whereas the
pseudo-inverse does. The target therefore inherits the donor's *entire* function,
patch projections included, and works in both directions:

  * coarse -> fine  (e.g. 8x8 -> 2x2): a finer model bootstrapped from a coarse one;
  * fine   -> coarse (e.g. 2x2 -> 8x8): a cheap coarse model that inherits the
    representation a well-trained fine model learned -- distillation by init.

The same mechanism doubles as a general "weight surgery" tool: any donor that
shares part of the target's architecture contributes every tensor whose name and
shape match (plus the PI-resized patch projections), and the rest keep their
fresh initialisation.

================================================================================
WHAT IT DOES
================================================================================
1. Builds the *target* model purely from its YAML config. The dataset-derived
   condition metadata is read from the donor checkpoint, so no dataset access is
   required.
2. Copies every donor tensor whose name AND shape match the target. The two patch
   projections (which differ in shape whenever the patch size changes) are
   PI-resized from the donor. Every other target tensor keeps its fresh
   initialisation. All sets (transferred / inherited / reinitialised / donor-only)
   are reported.
3. Writes a self-contained checkpoint with ``epoch = -1`` so the standard training
   loop resumes it at epoch 0 with a *fresh optimizer*. Point the target config's
   ``train.resume_from`` at the output file and train as usual.

================================================================================
USAGE
================================================================================
    # 1) build the inherited init (target arch = Base 8x8, donor = FinePatch2 2x2)
    python scripts/weight_inheritance.py \
        --donor  checkpoints/WaveDiT_CFM_Patch2/last.pth \
        --config configs/cfm_Base.yaml \
        --output checkpoints/WaveDiT_CFM_Base_inherited/last.pth

    # 2) train from it: a config whose run_name + train.resume_from point at the file
    bash train.sh configs/cfm_Base_inherited.yaml   # log should say 'Resumed at epoch 0'

================================================================================
CAVEATS
================================================================================
* Donor and target must share the transformer-body shape (depth, width, d_ff,
  attention types); only the patch size may differ. Body tensors that do not match
  fall back to a fresh initialisation.
* Changing the patch size changes the physical receptive field of neighborhood
  attention and the token-grid size, so a short adaptation phase is still
  expected. PI-resized projections start far ahead of random ones, not at zero
  cost.
* Only model weights are transferred; the optimizer restarts from scratch, which
  is the recommended behaviour when the patch size (and thus part of the parameter
  set) changes.
* ``--patch-init {flexivit,reinit}`` chooses how the two patch projections are set:
  ``flexivit`` PI-resizes them from the donor (learned), ``reinit`` leaves them at
  fresh init so they re-learn from scratch (the classic warm-start). ``--patch-out-init``
  overrides this for ``patch_out`` only (e.g. inherit the input embedding via PI-resize
  but re-learn the output head from scratch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow running as `python scripts/weight_inheritance.py ...` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wavedit import Config
from wavedit.models import build_model
from wavedit.training.checkpoint import strip_wrapper_prefixes

PATCH_IN = "patch_in.proj.weight"
PATCH_OUT = "patch_out.proj.weight"


def pi_resize_matrix(old_hw, new_hw, mode: str = "bilinear") -> torch.Tensor:
    """FlexiViT PI-resize matrix ``P`` of shape [new_h*new_w, old_h*old_w] such
    that ``new_filter = P @ old_filter`` reproduces a spatial patch filter at the
    new resolution in the least-squares sense (Beyer et al., 2023)."""
    oh, ow = old_hw
    nh, nw = new_hw
    rows = []
    for i in range(oh * ow):
        basis = torch.zeros(1, 1, oh, ow)
        basis.view(-1)[i] = 1.0
        up = F.interpolate(basis, size=(nh, nw), mode=mode, align_corners=False)
        rows.append(up.reshape(-1))
    resize = torch.stack(rows, dim=0)            # [old_hw, new_hw]
    return torch.linalg.pinv(resize)             # [new_hw, old_hw]


def resize_patch_in(w: torch.Tensor, old_p, new_p, P) -> torch.Tensor:
    """patch_in: [width, C*oph*opw] (channel-minor) -> [width, C*nph*npw]."""
    oph, opw = old_p
    nph, npw = new_p
    D = w.shape[0]
    C = w.shape[1] // (oph * opw)
    w = w.view(D, oph, opw, C).permute(0, 3, 1, 2).reshape(D * C, oph * opw)
    w = w @ P.t()                                # [D*C, nph*npw]
    return w.reshape(D, C, nph, npw).permute(0, 2, 3, 1).reshape(D, nph * npw * C)


def resize_patch_out(w: torch.Tensor, old_p, new_p, P) -> torch.Tensor:
    """patch_out: [C*oph*opw, width] (channel-minor) -> [C*nph*npw, width]."""
    oph, opw = old_p
    nph, npw = new_p
    D = w.shape[1]
    C = w.shape[0] // (oph * opw)
    w = w.view(oph, opw, C, D).permute(2, 3, 0, 1).reshape(C * D, oph * opw)
    w = w @ P.t()                                # [C*D, nph*npw]
    return w.reshape(C, D, nph, npw).permute(2, 3, 0, 1).reshape(nph * npw * C, D)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Warm-start a WaveDiT model from a pretrained checkpoint by "
                    "weight inheritance: shape-matched body transfer, with the two patch "
                    "projections either PI-resized (FlexiViT) or re-initialised.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--donor", required=True,
                   help="Path to the pretrained donor checkpoint (.pth).")
    p.add_argument("--config", required=True,
                   help="YAML config of the TARGET model to build.")
    p.add_argument("--output", required=True,
                   help="Where to write the inherited checkpoint "
                        "(typically the target run's last.pth).")
    p.add_argument("--patch-init", default="flexivit", choices=["flexivit", "reinit"],
                   help="How to set the two patch projections (patch_in/patch_out), which "
                        "change shape with the patch size. 'flexivit': PI-resize them from "
                        "the donor (learned transfer). 'reinit': leave them at fresh init so "
                        "they re-learn from scratch (the classic warm-start).")
    p.add_argument("--patch-out-init", default=None, choices=["flexivit", "reinit"],
                   help="Override --patch-init for patch_out only (defaults to --patch-init). "
                        "E.g. inherit the input embedding via PI-resize but re-learn the head.")
    p.add_argument("--interpolation", default="bilinear", choices=["bilinear", "bicubic"],
                   help="Spatial resize used to build the PI-resize operator (flexivit only).")
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
    target_patch = tuple(cfg.model.patch_size)
    init_in = args.patch_init
    init_out = args.patch_out_init or args.patch_init
    needs_pi = "flexivit" in (init_in, init_out)

    donor_model_cfg = donor.get("config", {}).get("model", {})
    donor_patch = tuple(donor_model_cfg["patch_size"]) if "patch_size" in donor_model_cfg else None
    if needs_pi and donor_patch is None:
        raise KeyError("Donor checkpoint has no embedded model.patch_size; needed for "
                       "FlexiViT PI-resize. Re-run with --patch-init reinit to skip it.")

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
    P = pi_resize_matrix(donor_patch, target_patch, mode=args.interpolation) if needs_pi else None

    # Shape-matched body transfer; patch projections follow --patch-init / --patch-out-init.
    transfer, inherited, reinit_patch, reinit_shape, reinit_absent = {}, {}, [], [], []
    for name, tensor in tgt.items():
        if name in src and src[name].shape == tensor.shape:
            transfer[name] = src[name]
        elif name.endswith(PATCH_IN) and name in src:
            if init_in == "flexivit":
                inherited[name] = resize_patch_in(src[name].float(), donor_patch, target_patch, P).to(tensor.dtype)
            else:
                reinit_patch.append(f"{name}  (patch_in, {init_in} -> fresh init)")
        elif name.endswith(PATCH_OUT) and name in src:
            if init_out == "flexivit":
                inherited[name] = resize_patch_out(src[name].float(), donor_patch, target_patch, P).to(tensor.dtype)
            else:
                reinit_patch.append(f"{name}  (patch_out, {init_out} -> fresh init)")
        elif name in src:
            reinit_shape.append(f"{name}  donor{tuple(src[name].shape)} != target{tuple(tensor.shape)}")
        else:
            reinit_absent.append(name)
    donor_only = [name for name in src if name not in tgt]

    missing, unexpected = model.load_state_dict({**transfer, **inherited}, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"

    print(f"Donor : {args.donor}  (patch {donor_patch})")
    print(f"Target: {args.config}  (patch {target_patch})")
    print(f"Patch projections: patch_in={init_in}, patch_out={init_out}")
    print(f"Transferred (name+shape match): {len(transfer)} / {len(tgt)} tensors")
    if inherited:
        print(f"Inherited via PI-resize ({args.interpolation}): {len(inherited)}")
        for name, t in inherited.items():
            print(f"    - {name}  donor{tuple(src[name].shape)} -> target{tuple(t.shape)}")
    if reinit_patch:
        print(f"Re-initialised (patch projection, fresh init): {len(reinit_patch)}")
        for line in reinit_patch:
            print(f"    - {line}")
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
        "note": f"weight inheritance from {Path(args.donor).name}: patch "
                f"{donor_patch} -> {target_patch} (body copied; "
                f"patch_in={init_in}, patch_out={init_out})",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Saved inherited checkpoint -> {out}")
    print(f"Trainable params: {n:,}")
    print("Set the target config's train.resume_from to this file and start training "
          "(the log should report 'Resumed at epoch 0').")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate brain-MRI samples from a trained WaveDiT checkpoint.

Two modes:

    # Fixed condition sets (N samples each)
    python scripts/generate.py CKPT.pth out/ specific --conditions "age=45.0" "age=70.0" --num-samples 10

    # Linearly interpolate one condition (1 sample per step)
    python scripts/generate.py CKPT.pth out/ linear --condition age --min 6 --max 95 --num 100

Checkpoints are self-contained: the model architecture and all condition metadata
are reconstructed from the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from wavedit import Config
from wavedit.generation import generate_samples, parse_condition_sets
from wavedit.models import build_model
from wavedit.training import load_model_weights
from wavedit.utils import get_logger, set_seed, setup_logging

logger = get_logger(__name__)

# Generation autocast precision, kept in sync with the training side (trainer.py).
_AUTOCAST_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a trained WaveDiT checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to a trained .pth checkpoint.")
    parser.add_argument("output_dir", type=str, help="Directory for the generated NIfTI files.")

    # Shared sampling options.
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-rescale", type=float, default=0.7)
    parser.add_argument("--num-flow-steps", type=int, default=None, help="Override the checkpoint's sampling steps.")
    parser.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"])
    parser.add_argument("--morpheus-scale", type=float, default=None, help="0 disables Morpheus guidance.")
    parser.add_argument("--save-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
                        help="Center-crop saved volumes to this size (default: full model output).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    specific = subparsers.add_parser("specific", help="Generate for explicit condition sets.")
    specific.add_argument("--conditions", nargs="+", required=True,
                          help='Condition sets, e.g. "age=45.0" "age=70.0 sex=1".')
    specific.add_argument("--num-samples", type=int, default=10, help="Samples per condition set.")

    linear = subparsers.add_parser("linear", help="Interpolate one numeric condition.")
    linear.add_argument("--condition", type=str, default="age", help="Condition to interpolate.")
    linear.add_argument("--min", type=float, required=True, dest="min_value")
    linear.add_argument("--max", type=float, required=True, dest="max_value")
    linear.add_argument("--num", type=int, default=100, dest="num_steps", help="Number of interpolation steps.")
    linear.add_argument("--fixed", nargs="*", default=[], help='Other fixed conditions, e.g. "sex=1".')

    args = parser.parse_args()
    if args.mode == "linear" and args.min_value >= args.max_value:
        parser.error("--min must be strictly less than --max")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_condition_list(args, condition_config) -> tuple[list[dict], int, bool]:
    """Return (conditions, samples_per_condition, group_by_condition) for the chosen mode."""
    if args.mode == "specific":
        conditions = parse_condition_sets(args.conditions, condition_config)
        if not conditions:
            raise SystemExit("No valid conditions parsed; nothing to generate.")
        return conditions, args.num_samples, True

    # linear
    key = args.condition.lower()
    if key not in condition_config:
        raise SystemExit(f"Condition '{key}' is not used by this model ({list(condition_config)}).")
    fixed = parse_condition_sets([" ".join(args.fixed)], condition_config)
    fixed = fixed[0] if fixed else {}
    values = np.linspace(args.min_value, args.max_value, args.num_steps)
    conditions = [{key: float(v), **fixed} for v in values]
    return conditions, 1, False


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, "wavedit_generate")
    set_seed(args.seed)

    device = resolve_device(args.device)
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    logger.info("Loading checkpoint %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    try:
        cfg = Config.from_dict(checkpoint["config"])
        condition_config = checkpoint["condition_config"]
        condition_ranges = checkpoint["condition_ranges"]
        categorical_maps = checkpoint["categorical_maps"]
        null_conditions = checkpoint["null_conditions"]
    except KeyError as exc:
        raise SystemExit(f"Checkpoint is missing required metadata key: {exc}")

    model = build_model(cfg, condition_config, condition_ranges, categorical_maps,
                        null_conditions, num_flow_steps=args.num_flow_steps).to(device)
    load_model_weights(model, checkpoint)
    model.eval()

    conditions, samples_per_condition, group = build_condition_list(args, condition_config)
    save_size = tuple(args.save_size) if args.save_size else cfg.data.image_size
    autocast_dtype = _AUTOCAST_DTYPES.get(cfg.precision, torch.bfloat16)

    generate_samples(
        model, conditions=conditions, num_samples_per_condition=samples_per_condition,
        output_dir=output_dir, save_size=save_size, model_output_size=cfg.data.image_size,
        cfg_scale=args.cfg_scale, sampler=args.sampler, morpheus_scale=args.morpheus_scale,
        cfg_rescale=args.cfg_rescale, group_by_condition=group, autocast_dtype=autocast_dtype,
    )
    logger.info("Generation finished.")


if __name__ == "__main__":
    main()

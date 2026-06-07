#!/usr/bin/env python3
"""Slim a WaveDiT checkpoint for release / inference.

Drops the optimiser, scheduler and scaler states (only needed to *resume* training)
while keeping the weights and the self-contained generation metadata, so the slim
file still works directly with ``scripts/generate.py``.

    python tools/slim_checkpoint.py checkpoints/run/best.pth checkpoints/run/best_slim.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from wavedit.training.checkpoint import strip_wrapper_prefixes
from wavedit.utils import get_logger

logger = get_logger(__name__)

# Everything needed to rebuild the model and generate; training-resume state is dropped.
KEEP_KEYS = (
    "epoch", "best_val_loss", "model_state_dict",
    "config", "condition_config", "condition_ranges",
    "categorical_maps", "cardinalities", "null_conditions",
)


def slim_checkpoint(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise SystemExit(f"Input checkpoint not found: {input_path}")
    logger.info("Loading %s", input_path)
    checkpoint = torch.load(input_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise SystemExit("Checkpoint has no 'model_state_dict'.")

    slim = {key: checkpoint[key] for key in KEEP_KEYS if key in checkpoint}
    slim["model_state_dict"] = strip_wrapper_prefixes(checkpoint["model_state_dict"])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(slim, output_path)

    before = os.path.getsize(input_path) / 1e6
    after = os.path.getsize(output_path) / 1e6
    logger.info("Saved %s (%.1f MB -> %.1f MB, -%.0f%%)",
                output_path, before, after, 100 * (before - after) / before if before else 0)


def main():
    parser = argparse.ArgumentParser(description="Slim a WaveDiT checkpoint for release/inference.")
    parser.add_argument("input_path", help="Path to the full training checkpoint (.pth).")
    parser.add_argument("output_path", help="Path for the slimmed checkpoint (.pth).")
    args = parser.parse_args()
    slim_checkpoint(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

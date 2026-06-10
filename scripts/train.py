#!/usr/bin/env python3
"""Train WaveDiT from a single YAML config.

    python scripts/train.py configs/cfm.yaml

Everything (data, model, optimisation, logging) is described by the config; this
script just wires the pieces together.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow `python scripts/train.py ...` from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import wandb
from torch.utils.data import DataLoader

from wavedit import Config
from wavedit.data import build_datasets, collate_fn
from wavedit.evaluation import prepare_real_reference, visualize_condition_sweep, visualize_generation
from wavedit.generation import generate_samples, parse_condition_sets
from wavedit.models import build_model
from wavedit.training import Trainer
from wavedit.utils import get_logger, set_seed, setup_logging

logger = get_logger(__name__)

# Autocast precision, kept in sync with trainer.py and scripts/generate.py.
_AUTOCAST_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs per dataloader worker.

    Without this, forked workers share the parent's NumPy seed, so the NumPy-backed
    MONAI augmentations produce correlated/duplicated randomness across workers.
    ``torch.initial_seed()`` is already unique per worker (derived from the loader's
    base generator), so we derive the other RNGs from it.
    """
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_loaders(bundle, cfg: Config):
    # Seed shuffling and the workers so augmentation RNG is reproducible across runs.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)
    worker_init = seed_worker if cfg.data.num_workers > 0 else None

    train_loader = DataLoader(
        bundle.train, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=worker_init, generator=loader_generator,
    )
    val_loader = None
    if bundle.val is not None and len(bundle.val) > 0:
        val_loader = DataLoader(
            bundle.val, batch_size=cfg.train.batch_size, shuffle=False,
            num_workers=cfg.data.num_workers, pin_memory=True,
            collate_fn=collate_fn, persistent_workers=cfg.data.num_workers > 0,
            worker_init_fn=worker_init,
        )
    return train_loader, val_loader


def make_visualizer(model, val_loader, cfg: Config):
    """Per-epoch W&B callback: one real reference at start, then one synthetic volume per epoch."""
    autocast_dtype = _AUTOCAST_DTYPES.get(cfg.precision, torch.bfloat16)
    cache: dict = {}

    def visualize(epoch: int) -> None:
        if cache.get("real_recon") is None:
            cache["real_recon"] = prepare_real_reference(model, val_loader, use_wandb=True)
        visualize_generation(model, cache["real_recon"],
                             sampler=cfg.sampling.sampler, cfg_rescale=cfg.sampling.cfg_rescale,
                             epoch=epoch, use_wandb=True, autocast_dtype=autocast_dtype)

    return visualize


def run_post_training_generation(model, cfg: Config, run_dir: Path):
    gen = cfg.post_train_generation
    conditions = parse_condition_sets(gen.conditions, model.condition_config)
    if not conditions:
        logger.warning("No valid post-training conditions parsed; skipping generation.")
        return
    generate_samples(
        model, conditions=conditions, num_samples_per_condition=gen.num_samples,
        output_dir=run_dir / "post_train_samples", save_size=gen.save_size,
        model_output_size=cfg.data.image_size, cfg_scale=gen.cfg_scale,
        sampler=cfg.sampling.sampler, morpheus_scale=cfg.model.morpheus_scale,
        cfg_rescale=cfg.sampling.cfg_rescale,
        autocast_dtype=_AUTOCAST_DTYPES.get(cfg.precision, torch.bfloat16),
    )


def main():
    parser = argparse.ArgumentParser(description="Train WaveDiT from a YAML config.")
    parser.add_argument("config", type=str, help="Path to the experiment YAML config.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_dir = Path(cfg.logging.checkpoint_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir, cfg.run_name)
    logger.info("Starting run '%s' -> %s", cfg.run_name, run_dir)
    shutil.copy(args.config, run_dir / "config.yaml")  # provenance

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    if cfg.logging.wandb:
        try:
            wandb.init(project=cfg.logging.wandb_project, entity=cfg.logging.wandb_entity,
                       name=cfg.run_name, config=cfg.to_dict())
            # NOTE: do not use wandb.define_metric() here — as of 2026-06 it silently
            # breaks server-side history ingestion (rows never appear, only summary).
            # Stepless wandb.log with 'epoch'/'global_step' as plain fields works; set
            # the chart X axis to 'epoch' in the workspace settings instead.
        except Exception as exc:  # noqa: BLE001 - never let logging setup abort a run
            logger.warning("Could not initialise W&B (%s); continuing without it.", exc)
            cfg.logging.wandb = False

    bundle = build_datasets(cfg.data, cfg.seed)
    train_loader, val_loader = build_loaders(bundle, cfg)
    logger.info("Train batches: %d | Val batches: %s",
                len(train_loader), len(val_loader) if val_loader else "none")

    model = build_model(cfg, bundle.condition_config, bundle.condition_ranges,
                        bundle.categorical_maps, bundle.null_conditions).to(device)

    checkpoint_metadata = {
        "config": cfg.to_dict(),
        "condition_config": bundle.condition_config,
        "condition_ranges": bundle.condition_ranges,
        "categorical_maps": bundle.categorical_maps,
        "cardinalities": bundle.cardinalities,
        "null_conditions": bundle.null_conditions,
    }
    visualizer = make_visualizer(model, val_loader, cfg) if (cfg.logging.wandb and val_loader) else None

    trainer = Trainer(model, train_loader, val_loader, cfg, run_dir, checkpoint_metadata, visualizer)
    trainer.fit()
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # One condition sweep on the final (best) weights instead of one per epoch.
    if cfg.logging.wandb and model.condition_config:
        sweep_key = "age" if "age" in model.condition_config else next(iter(model.condition_config))
        visualize_condition_sweep(model, cfg.data.image_size, sweep_key, num_values=5,
                                  use_wandb=True,
                                  autocast_dtype=_AUTOCAST_DTYPES.get(cfg.precision, torch.bfloat16))

    if cfg.post_train_generation.enabled:
        logger.info("Running post-training sample generation.")
        run_post_training_generation(model, cfg, run_dir)

    if cfg.logging.wandb:
        wandb.finish()
    logger.info("Run '%s' finished.", cfg.run_name)


if __name__ == "__main__":
    main()

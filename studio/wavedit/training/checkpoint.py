"""Checkpoint serialisation.

Checkpoints are **self-contained**: alongside the weights they store the full
resolved config and the dataset-derived condition metadata, so generation can
rebuild an identical model from a single ``.pth`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..utils import get_logger

logger = get_logger(__name__)

# Wrapper prefixes added by torch.compile / DistributedDataParallel.
_WRAPPER_PREFIXES = ("_orig_mod.", "module.")


def strip_wrapper_prefixes(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_model_weights(model: torch.nn.Module, checkpoint: dict) -> None:
    """Load weights from a checkpoint into ``model`` (non-strict, prefix-tolerant)."""
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint has no 'model_state_dict'.")
    state = strip_wrapper_prefixes(checkpoint["model_state_dict"])
    target = getattr(model, "_orig_mod", model)
    missing, unexpected = target.load_state_dict(state, strict=False)
    logger.info("Loaded model weights (missing=%d, unexpected=%d).", len(missing), len(unexpected))


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
    epochs_without_improvement: int,
    metadata: dict[str, Any],
) -> None:
    """Write a self-contained checkpoint. ``metadata`` carries config + condition info."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
        **metadata,
    }
    torch.save(payload, path)


def resume_training(path: str | Path, model, optimizer, scheduler, scaler, device) -> tuple[int, float, int]:
    """Restore model/optimizer/scheduler/scaler state. Returns (start_epoch, best_val_loss, patience)."""
    logger.info("Resuming training from %s", path)
    checkpoint = torch.load(path, map_location=device)

    load_model_weights(model, checkpoint)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore scheduler state: %s", exc)
    if scaler.is_enabled() and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    patience = checkpoint.get("epochs_without_improvement", 0)
    logger.info("Resumed at epoch %d (best_val_loss=%.6f).", start_epoch, best_val_loss)
    return start_epoch, best_val_loss, patience

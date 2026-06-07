"""Training loop for WaveDiT."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..config import Config
from ..utils import get_logger
from .checkpoint import load_model_weights, resume_training, save_checkpoint

logger = get_logger(__name__)

_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class Trainer:
    """Drives optimisation, validation, checkpointing and optional visualisation.

    ``visualize_fn(epoch)`` is an optional callback invoked at the end of each epoch
    (when W&B is enabled and a validation set exists) so the training loop stays
    decoupled from the evaluation/plotting code.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        cfg: Config,
        checkpoint_dir: str | Path,
        checkpoint_metadata: dict,
        visualize_fn: Callable[[int], None] | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = checkpoint_metadata
        self.visualize_fn = visualize_fn

        self.device = next(model.parameters()).device
        self.use_wandb = cfg.logging.wandb

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.train.epochs, eta_min=1e-7)

        # Autocast for fp16/bf16 on CUDA; loss scaling only for fp16 (bf16 has the
        # dynamic range to train without it).
        self.autocast_dtype = _AUTOCAST_DTYPES[cfg.precision]
        self.use_amp = self.device.type == "cuda" and cfg.precision in ("fp16", "bf16")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.device.type == "cuda" and cfg.precision == "fp16"))

        self.best_path = self.checkpoint_dir / "best.pth"
        self.last_path = self.checkpoint_dir / "last.pth"
        self.global_step = 0

    # ------------------------------------------------------------------ #
    def fit(self) -> float:
        start_epoch, best_val_loss, epochs_without_improvement = 0, float("inf"), 0
        resume_from = self.cfg.train.resume_from
        if resume_from and Path(resume_from).is_file():
            start_epoch, best_val_loss, epochs_without_improvement = resume_training(
                resume_from, self.model, self.optimizer, self.scheduler, self.scaler, self.device
            )
        elif resume_from:
            logger.warning("resume_from '%s' not found; starting from scratch.", resume_from)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Training %d epochs from epoch %d. Trainable params: %s.",
                    self.cfg.train.epochs, start_epoch, f"{total_params:,}")
        self.global_step = start_epoch * max(1, len(self.train_loader))

        for epoch in range(start_epoch, self.cfg.train.epochs):
            epoch_start = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate(epoch)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d [%.1fs] | train_loss=%.6f val_loss=%.6f lr=%.3e",
                epoch + 1, self.cfg.train.epochs, elapsed, train_loss, val_loss, current_lr,
            )
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss if self.val_loader is not None else None,
                    "learning_rate": current_lr,
                    "epoch_time_sec": elapsed,
                }, step=self.global_step)
                if self.val_loader is not None and self.visualize_fn is not None:
                    self._run_visualization(epoch)

            best_val_loss, epochs_without_improvement, should_stop = self._save_and_track(
                epoch, val_loss, best_val_loss, epochs_without_improvement
            )
            if should_stop:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

        self._restore_best()
        return best_val_loss

    # ------------------------------------------------------------------ #
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, total_samples, dropped_batches = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False)
        for batch in pbar:
            if batch is None:
                dropped_batches += 1
                continue
            images, conditions = batch
            images = images.to(self.device)
            conditions = {k: v.to(self.device) for k, v in conditions.items()}
            batch_size = images.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                loss = self.model.loss(images, conditions, self.global_step, use_wandb_logging=self.use_wandb)

            if not torch.isfinite(loss):
                logger.error("Epoch %d: non-finite loss (%s); skipping step.", epoch + 1, loss.item())
                continue

            self.scaler.scale(loss).backward()
            grad_norm = torch.tensor(0.0)
            if self.cfg.train.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_step += 1
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pbar.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm.item():.3f}")
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({"train/grad_norm": grad_norm.item()}, step=self.global_step)

        if dropped_batches:
            logger.warning("Epoch %d: dropped %d batch(es) with no loadable samples.", epoch + 1, dropped_batches)
        return total_loss / total_samples if total_samples else float("inf")

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        if self.val_loader is None:
            return float("inf")
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [val]", leave=False)
        for batch in pbar:
            if batch is None:
                continue
            images, conditions = batch
            images = images.to(self.device)
            conditions = {k: v.to(self.device) for k, v in conditions.items()}
            with torch.amp.autocast(device_type=self.device.type, dtype=self.autocast_dtype, enabled=self.use_amp):
                loss = self.model.loss(images, conditions, self.global_step, use_wandb_logging=False)
            if not torch.isfinite(loss):
                continue
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        return total_loss / total_samples if total_samples else float("inf")

    def _run_visualization(self, epoch: int) -> None:
        try:
            self.visualize_fn(epoch + 1)
        except Exception as exc:  # noqa: BLE001 - viz must never crash training
            logger.error("Visualization failed at epoch %d: %s", epoch + 1, exc, exc_info=True)

    def _save_and_track(self, epoch, val_loss, best_val_loss, epochs_without_improvement):
        def write(path, best):
            save_checkpoint(
                path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, epoch=epoch, best_val_loss=best,
                epochs_without_improvement=epochs_without_improvement, metadata=self.metadata,
            )

        write(self.last_path, best_val_loss)

        # No validation set -> the latest model is the best we have.
        if self.val_loader is None:
            write(self.best_path, best_val_loss)
            return best_val_loss, epochs_without_improvement, False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            write(self.best_path, best_val_loss)
            logger.info("New best model (val_loss=%.6f).", best_val_loss)
        else:
            epochs_without_improvement += 1
            patience = self.cfg.train.early_stop_patience
            if patience > 0 and epochs_without_improvement >= patience:
                return best_val_loss, epochs_without_improvement, True
        return best_val_loss, epochs_without_improvement, False

    def _restore_best(self) -> None:
        path = self.best_path if self.best_path.exists() else self.last_path
        if path.exists():
            logger.info("Restoring best weights from %s for post-training tasks.", path)
            load_model_weights(self.model, torch.load(path, map_location=self.device))

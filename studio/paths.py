"""Data directory layout for WaveDiT Studio.

Default data dir is platform specific (macOS: ~/Library/Application Support/WaveDiT Studio)
and can be overridden with WAVEDIT_STUDIO_DATA_DIR. The checkpoints dir can be pointed at an
existing directory with WAVEDIT_STUDIO_CKPT_DIR. Every accessor creates its directory on
first call.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """Return the app data directory, creating it if needed."""
    override = os.environ.get("WAVEDIT_STUDIO_DATA_DIR")
    if override:
        path = Path(override).expanduser()
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "WaveDiT Studio"
    else:
        path = Path.home() / ".local" / "share" / "wavedit-studio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir() -> Path:
    """Return the model checkpoints directory.

    If WAVEDIT_STUDIO_CKPT_DIR is set it is returned as-is (created only if its parent
    already exists); otherwise <data dir>/checkpoints is used.
    """
    override = os.environ.get("WAVEDIT_STUDIO_CKPT_DIR")
    if override:
        path = Path(override).expanduser()
        path.mkdir(exist_ok=True)
        return path
    path = data_dir() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_dir() -> Path:
    """Return the library directory holding saved generations."""
    path = data_dir() / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exports_dir() -> Path:
    """Return the exports directory (headless fallback for /api/export)."""
    path = data_dir() / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path

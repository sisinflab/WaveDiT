"""Logging helpers shared across the WaveDiT codebase."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str | Path, filename_prefix: str = "wavedit") -> Path:
    """Configure root logging to write to both a timestamped file and the console.

    Safe to call more than once per process: existing handlers are replaced so a
    new run logs to its own file rather than appending to the previous one.

    Returns the path of the created log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{filename_prefix}_{timestamp}.log"

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger(__name__).info("Logging initialised -> %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper kept for a single import site)."""
    return logging.getLogger(name)

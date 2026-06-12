"""Persisted user settings for WaveDiT Studio.

Thread-safe, JSON-backed store at data_dir()/settings.json. Writes are atomic
(temp file + os.replace) and a corrupted file silently recovers to defaults.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any

from .paths import data_dir

_DEFAULTS: dict[str, Any] = {
    "precision": "auto",
    "bf16_ok": None,
    "default_model": "WaveDiT-Base.pth",
    "calibration": {},
    "onboarding_done": False,
}


class Settings:
    """Thread-safe settings store persisted as JSON."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = data_dir() / "settings.json"
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        data = copy.deepcopy(_DEFAULTS)
        try:
            with open(self._path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                data.update(stored)
        except (OSError, ValueError):
            pass
        return data

    def _save_locked(self) -> None:
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def get(self) -> dict[str, Any]:
        """Return a deep copy of the current settings."""
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, partial: dict[str, Any]) -> dict[str, Any]:
        """Merge a partial dict into the settings, persist, and return the full copy."""
        with self._lock:
            self._data.update(copy.deepcopy(partial))
            self._save_locked()
            return copy.deepcopy(self._data)


settings = Settings()

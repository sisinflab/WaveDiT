"""On-disk library of generated volumes for WaveDiT Studio.

Each item lives at <data dir>/library/<id>/ with volume.nii.gz (float32,
intensities in [0, 1]), thumb.png (mid-axial slice) and meta.json (full
provenance). meta.json is written last and atomically, so its presence
marks a complete entry; readers skip anything else.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import minipng
from .paths import library_dir

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_id(item_id: str) -> str:
    """Validate an id before any path construction."""
    if not isinstance(item_id, str) or not _ID_RE.match(item_id):
        raise ValueError(f"invalid library id: {item_id!r}")
    return item_id


def _created_key(item: dict) -> float:
    """Sort key tolerant of missing or malformed created fields."""
    try:
        return float(item.get("created", 0.0))
    except (TypeError, ValueError):
        return 0.0


class Library:
    """Thread-safe store of generations: volume.nii.gz + thumb.png + meta.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # -- write -----------------------------------------------------------

    def save(self, arr_f32: np.ndarray, meta: dict) -> dict:
        """Persist a generated volume and its provenance; returns the item dict."""
        arr = np.asarray(arr_f32, dtype=np.float32)
        created = float(meta.get("created") or time.time())
        with self._lock:
            item_id = self._new_id(meta.get("seed", 0), created)
            item = dict(meta)
            item["id"] = item_id
            item["created"] = created
            item["vol_url"] = f"/volumes/{item_id}.nii.gz"
            item["thumb_url"] = f"/thumbs/{item_id}.png"

            entry = library_dir() / item_id
            entry.mkdir(parents=True, exist_ok=True)
            self._write_volume(arr, entry / "volume.nii.gz")
            (entry / "thumb.png").write_bytes(self._thumb_bytes(arr))
            tmp = entry / "meta.json.tmp"
            tmp.write_text(json.dumps(item, indent=2), encoding="utf-8")
            os.replace(tmp, entry / "meta.json")
        return item

    @staticmethod
    def _new_id(seed: object, created: float) -> str:
        """YYYYMMDD-HHMMSS-<seed>-<4 hex>, unique within the library dir."""
        try:
            seed_i = int(seed)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            seed_i = 0
        stamp = datetime.fromtimestamp(created).strftime("%Y%m%d-%H%M%S")
        root = library_dir()
        while True:
            item_id = f"{stamp}-{seed_i}-{secrets.token_hex(2)}"
            _check_id(item_id)
            if not (root / item_id).exists():
                return item_id

    @staticmethod
    def _write_volume(arr: np.ndarray, path: Path) -> None:
        """Write the float32 volume as gzipped NIfTI with an identity affine."""
        import nibabel as nib  # lazy: keeps app cold start fast

        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))

    @staticmethod
    def _thumb_bytes(arr: np.ndarray) -> bytes:
        """Mid-axial slice, rotated like the Space preview, as 8-bit gray PNG."""
        mid = np.rot90(arr[arr.shape[0] // 2, :, :])
        img = np.clip(mid, 0.0, 1.0) * 255.0
        return minipng.encode_gray_png(img.astype(np.uint8))

    # -- read ------------------------------------------------------------

    def list(self) -> list[dict]:
        """All complete items, newest first; partial entries are skipped."""
        items: list[dict] = []
        for entry in library_dir().iterdir():
            if not entry.is_dir() or not _ID_RE.match(entry.name):
                continue
            try:
                item = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue  # partially written, partially deleted or corrupt
            if isinstance(item, dict):
                items.append(item)
        items.sort(key=_created_key, reverse=True)
        return items

    def get(self, item_id: str) -> dict | None:
        """The item dict for an id, or None if absent or unreadable."""
        try:
            _check_id(item_id)
        except ValueError:
            return None
        try:
            item = json.loads(
                (library_dir() / item_id / "meta.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return item if isinstance(item, dict) else None

    def count(self) -> int:
        """Number of complete, readable entries (consistent with list())."""
        return len(self.list())

    # -- paths -----------------------------------------------------------

    def volume_path(self, item_id: str) -> Path:
        _check_id(item_id)
        return library_dir() / item_id / "volume.nii.gz"

    def thumb_path(self, item_id: str) -> Path:
        _check_id(item_id)
        return library_dir() / item_id / "thumb.png"

    # -- delete ----------------------------------------------------------

    def delete(self, item_id: str) -> bool:
        """Remove an entry directory; True if it existed and is gone."""
        try:
            _check_id(item_id)
        except ValueError:
            return False
        entry = library_dir() / item_id
        with self._lock:
            if not entry.is_dir():
                return False
            shutil.rmtree(entry, ignore_errors=True)
            return not entry.exists()


library = Library()

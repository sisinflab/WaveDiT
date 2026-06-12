"""Hugging Face weights manager for WaveDiT Studio.

Lists the *.pth checkpoints of the public repo, streams downloads with
throttled progress callbacks, and falls back to a purely local listing
when offline so the UI always has something actionable to render.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from .paths import checkpoints_dir

HF_REPO = "danesed/WaveDiT"

_RESOLVE_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/"
_KNOWN_FILES = (
    "WaveDiT-Base.pth",
    "WaveDiT-FinePatch.pth",
    "WaveDiT-Deep.pth",
    "WaveDiT-Wide.pth",
)
_CACHE_TTL_S = 300.0
_FAIL_TTL_S = 30.0  # negative cache: do not re-hit the network for every /api/state
_CHUNK_BYTES = 1024 * 1024
_PROGRESS_MIN_INTERVAL_S = 0.25  # at most ~4 progress callbacks per second
_HTTP_TIMEOUT_S = 30
_LIST_TIMEOUT_S = 10  # listing blocks /api/state, so keep its worst case short
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _https_context() -> ssl.SSLContext:
    """Trust store that survives PyInstaller freezing.

    The frozen libssl's baked-in OPENSSLDIR may not exist on the end user's Mac,
    so prefer certifi's bundled CA file when available.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _label(file: str) -> str:
    """Human label for a checkpoint filename (WaveDiT-Base.pth -> Base)."""
    stem = file[:-4] if file.endswith(".pth") else file
    return stem.removeprefix("WaveDiT-") or stem


def _check_name(file: str) -> str:
    """Reject anything that could escape the checkpoints directory."""
    if not isinstance(file, str) or ".." in file or not _SAFE_NAME_RE.match(file):
        raise ValueError(f"invalid checkpoint name: {file!r}")
    return file


def _ordered(names: Iterable[str]) -> list[str]:
    """Known files first in canonical order, then extras alphabetically."""
    pool = set(names)
    known = [n for n in _KNOWN_FILES if n in pool]
    extra = sorted(n for n in pool if n not in _KNOWN_FILES)
    return known + extra


def _short_reason(exc: Exception) -> str:
    """One-line human reason for a failed download."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc) or exc.__class__.__name__


class WeightsManager:
    """Remote listing, streamed download and deletion of model checkpoints."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._downloading: set[str] = set()
        self._remote: dict[str, int | None] | None = None  # file -> size bytes
        self._remote_at: float = 0.0
        self._failed_at: float = 0.0
        self._purge_orphan_parts()

    def _purge_orphan_parts(self) -> None:
        """Remove stale .part files left behind by a killed process."""
        with self._lock:
            active = set(self._downloading)
        try:
            for part in checkpoints_dir().glob("*.part"):
                if part.name[: -len(".part")] not in active:
                    part.unlink(missing_ok=True)
        except OSError:
            pass

    # -- listing ---------------------------------------------------------

    def list(self, refresh: bool = False) -> list[dict]:
        """Merged remote + local view: [{file, label, size_mb, downloaded, downloading}]."""
        remote = self._remote_sizes(refresh=refresh)
        ckpt_dir = checkpoints_dir()
        local = {p.name for p in ckpt_dir.glob("*.pth") if p.is_file()}
        if remote is None:
            # Offline: local files plus the known catalog so the UI can still act.
            names = _ordered(local | set(_KNOWN_FILES))
            sizes: dict[str, int | None] = {}
        else:
            names = _ordered(set(remote) | local)
            sizes = remote
        with self._lock:
            downloading = set(self._downloading)
        items: list[dict] = []
        for name in names:
            path = ckpt_dir / name
            downloaded = path.is_file()
            size_b = sizes.get(name)
            if size_b is None and downloaded:
                try:
                    size_b = path.stat().st_size
                except OSError:
                    size_b = None
            items.append(
                {
                    "file": name,
                    "label": _label(name),
                    "size_mb": round(size_b / (1024 * 1024), 1) if size_b else None,
                    "downloaded": downloaded,
                    "downloading": name in downloading,
                }
            )
        return items

    def _remote_sizes(self, refresh: bool) -> dict[str, int | None] | None:
        """Remote {file: size} with a 300 s TTL cache; None when fully offline.

        Failures are negatively cached for 30 s so an offline machine does not
        block every /api/state call on a fresh network timeout.
        """
        with self._lock:
            fresh = self._remote is not None and (time.monotonic() - self._remote_at) < _CACHE_TTL_S
            if fresh and not refresh:
                return dict(self._remote or {})
            recently_failed = (time.monotonic() - self._failed_at) < _FAIL_TTL_S
            if recently_failed and not refresh:
                return dict(self._remote) if self._remote is not None else None
        fetched = self._fetch_remote()
        with self._lock:
            if fetched is not None:
                self._remote = fetched
                self._remote_at = time.monotonic()
            else:
                self._failed_at = time.monotonic()
            # On fetch failure a stale cache still beats the offline fallback.
            return dict(self._remote) if self._remote is not None else None

    @staticmethod
    def _fetch_remote() -> dict[str, int | None] | None:
        """Query the Hub tree endpoint for *.pth files and sizes; None on any failure.

        One bounded urllib call (files and sizes together) instead of huggingface_hub,
        whose HfApi requests carry no timeout and would stall /api/state.
        """
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/models/{HF_REPO}/tree/main",
                headers={"User-Agent": "wavedit-studio"},
            )
            with urllib.request.urlopen(
                req, timeout=_LIST_TIMEOUT_S, context=_https_context()
            ) as resp:
                entries = json.load(resp)
            # Top-level *.pth only: nested paths (e.g. _prerelease_checkpoints/)
            # cannot be stored flat in the checkpoints dir.
            return {
                e["path"]: (int(e["size"]) if e.get("size") else None)
                for e in entries
                if e.get("type") == "file" and e["path"].endswith(".pth") and "/" not in e["path"]
            }
        except Exception:
            return None

    # -- download --------------------------------------------------------

    def download(self, file: str, progress_cb: Callable[[dict], None]) -> Path:
        """Stream <file> from the Hub to the checkpoints dir; returns the final path."""
        _check_name(file)
        ckpt_dir = checkpoints_dir()
        target = ckpt_dir / file
        with self._lock:
            if file in self._downloading:
                raise RuntimeError(f"{file} is already downloading")
            self._downloading.add(file)
        part = ckpt_dir / (file + ".part")
        try:
            if target.is_file():
                return target
            url = _RESOLVE_BASE + urllib.parse.quote(file)
            req = urllib.request.Request(url, headers={"User-Agent": "wavedit-studio"})
            started = time.monotonic()
            last_emit = 0.0
            done = 0
            with urllib.request.urlopen(
                req, timeout=_HTTP_TIMEOUT_S, context=_https_context()
            ) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                with open(part, "wb") as out:
                    while True:
                        chunk = resp.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_emit >= _PROGRESS_MIN_INTERVAL_S:
                            last_emit = now
                            self._emit(progress_cb, done, total, started)
            os.replace(part, target)
            self._emit(progress_cb, done, max(total, done), started)
            return target
        except (OSError, http.client.HTTPException) as exc:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed for {file}: {_short_reason(exc)}") from exc
        finally:
            with self._lock:
                self._downloading.discard(file)

    @staticmethod
    def _emit(
        progress_cb: Callable[[dict], None], done_b: int, total_b: int, started: float
    ) -> None:
        """Publish one progress snapshot; callback errors never abort the download."""
        elapsed = max(time.monotonic() - started, 1e-6)
        mb_done = done_b / (1024 * 1024)
        mb_total = total_b / (1024 * 1024)
        try:
            progress_cb(
                {
                    "mb_done": round(mb_done, 1),
                    "mb_total": round(mb_total, 1),
                    "pct": round(min(done_b / total_b * 100.0, 100.0), 1) if total_b else 0.0,
                    "speed_mbps": round(mb_done / elapsed, 1),  # megabytes per second
                }
            )
        except Exception:
            pass

    # -- local state -----------------------------------------------------

    def delete(self, file: str) -> bool:
        """Remove a downloaded checkpoint; refuses while it is downloading."""
        _check_name(file)
        with self._lock:
            if file in self._downloading:
                return False
        ckpt_dir = checkpoints_dir()
        (ckpt_dir / (file + ".part")).unlink(missing_ok=True)
        path = ckpt_dir / file
        if path.is_file():
            path.unlink()
            return True
        return False

    def local_path(self, file: str) -> Path | None:
        """Path of a downloaded checkpoint, or None if absent or invalid."""
        try:
            _check_name(file)
        except ValueError:
            return None
        path = checkpoints_dir() / file
        return path if path.is_file() else None

    def is_downloaded(self, file: str) -> bool:
        return self.local_path(file) is not None


weights = WeightsManager()

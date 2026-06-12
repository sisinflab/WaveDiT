"""Chip, RAM, OS and torch device detection for WaveDiT Studio.

torch is imported lazily so this module can be loaded before main.py sets the
torch-related environment variables.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def _is_available(name: str) -> bool:
    """Return True if the torch device type is usable on this machine."""
    import torch

    try:
        if name == "mps":
            backend = getattr(torch.backends, "mps", None)
            return backend is not None and backend.is_available()
        if name == "cuda":
            return torch.cuda.is_available()
        return name == "cpu"
    except Exception:
        return False


def pick_device() -> torch.device:
    """Pick the torch device: WAVEDIT_STUDIO_DEVICE if set and available, else mps > cuda > cpu."""
    import torch

    forced = os.environ.get("WAVEDIT_STUDIO_DEVICE", "").strip().lower()
    if forced:
        try:
            dev = torch.device(forced)
        except (RuntimeError, ValueError):
            dev = None
        if dev is not None and _is_available(dev.type):
            return dev
    for name in ("mps", "cuda"):
        if _is_available(name):
            return torch.device(name)
    return torch.device("cpu")


def _chip() -> str:
    """Best-effort CPU / SoC name. Never raises."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            name = out.stdout.strip()
            if name:
                return name
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    try:
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def _ram_gb() -> float:
    """Total physical RAM in GiB, 0.0 if it cannot be determined."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            return round(int(out.stdout.strip()) / 2**30, 1)
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 1)
    except Exception:
        return 0.0


def get_sysinfo() -> dict[str, Any]:
    """Snapshot of the host: os, os_version, chip, ram_gb, python, torch, device."""
    info: dict[str, Any] = {
        "os": "unknown",
        "os_version": "unknown",
        "chip": _chip(),
        "ram_gb": _ram_gb(),
        "python": "unknown",
        "torch": "unknown",
        "device": "unknown",
    }
    try:
        info["python"] = platform.python_version()
        if sys.platform == "darwin":
            info["os"] = "macOS"
            info["os_version"] = platform.mac_ver()[0] or platform.release() or "unknown"
        else:
            info["os"] = platform.system() or "unknown"
            info["os_version"] = platform.release() or "unknown"
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = pick_device().type
    except Exception:
        pass
    return info

"""WaveDiT Studio entry point.

Starts the local backend server in a daemon thread, then opens a native
pywebview (WKWebView) window pointed at it. Falls back to headless mode when
WAVEDIT_STUDIO_HEADLESS=1 or when pywebview is unavailable. Produces synthetic
research images only; not a medical device.
"""
from __future__ import annotations

import os
import sys

# Must be set before torch (or anything that imports torch) is loaded.
os.environ.setdefault("WAVEDIT_NA_BACKEND", "torch")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The vendored HDiT wraps its hot kernels in torch.compile when this flag is "1"
# (the default). Inside a frozen PyInstaller app Dynamo cannot see the source files
# (modules live in the PYZ archive) and the Inductor MPS backend is experimental,
# so compilation can only stall or fail here: force eager.
os.environ.setdefault("K_DIFFUSION_USE_COMPILE", "0")

# The frozen libssl's baked-in OPENSSLDIR may not exist on the end user's Mac;
# point every OpenSSL consumer in the bundle at certifi's CA file instead.
try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

from pathlib import Path


def _studio_dir() -> Path:
    """Locate the studio package dir in both source and frozen (PyInstaller) runs."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return Path(base) / "studio"
    return Path(__file__).resolve().parent


_STUDIO_DIR = _studio_dir()
# The vendored top-level "wavedit" package (studio/wavedit) must win over any repo copy.
_sd = str(_STUDIO_DIR)
if _sd in sys.path:
    sys.path.remove(_sd)
sys.path.insert(0, _sd)
# Make "studio" itself importable when this file runs as a plain script.
_parent = str(_STUDIO_DIR.parent)
if _parent not in sys.path:
    sys.path.append(_parent)

import argparse
import json
import shutil
import time
from collections.abc import Callable
from typing import Any


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wavedit-studio",
        description="WaveDiT Studio: age-conditioned synthetic 3D brain MRI (research only).",
    )
    parser.add_argument(
        "--selfcheck", action="store_true", help="print versions and device as JSON, then exit"
    )
    parser.add_argument("--port", type=int, default=None, help="fixed server port")
    return parser.parse_args(argv)


def _selfcheck() -> None:
    """Print environment info as JSON and exit 0 (used by build.sh smoke test)."""
    import torch

    from studio.sysinfo import pick_device

    try:
        from studio import __version__ as app_version
    except Exception:
        app_version = "0.0.0"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    info = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "device": str(pick_device()),
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "app": app_version,
    }
    print(json.dumps(info), flush=True)
    sys.exit(0)


def _run_headless(url: str) -> None:
    print("", flush=True)
    print("WaveDiT Studio is running in headless mode.", flush=True)
    print(f"  Open in a browser: {url}", flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[studio] stopped.", flush=True)


def _make_export_handler(webview: Any) -> Callable[[str, str], str | None]:
    """Build the native save-dialog export handler installed on the server module."""

    def export_via_dialog(src_path: str, suggested_name: str) -> str | None:
        windows = getattr(webview, "windows", None)
        if not windows:
            return None
        # pywebview 6 renamed the dialog enum; keep the old constant as fallback.
        file_dialog = getattr(webview, "FileDialog", None)
        dialog_kind = file_dialog.SAVE if file_dialog else webview.SAVE_DIALOG
        try:
            result = windows[0].create_file_dialog(
                dialog_kind, save_filename=suggested_name
            )
        except Exception as exc:
            print(f"[studio] save dialog failed: {exc}", flush=True)
            return None
        if not result:
            return None
        dest = result[0] if isinstance(result, (list, tuple)) else result
        if not dest:
            return None
        shutil.copyfile(src_path, str(dest))
        return str(dest)

    return export_via_dialog


def _make_import_handler(webview: Any) -> Callable[[], str | None]:
    """Build the native open-dialog import handler installed on the server module."""

    def import_via_dialog() -> str | None:
        windows = getattr(webview, "windows", None)
        if not windows:
            return None
        file_dialog = getattr(webview, "FileDialog", None)
        dialog_kind = file_dialog.OPEN if file_dialog else webview.OPEN_DIALOG
        try:
            result = windows[0].create_file_dialog(
                dialog_kind,
                allow_multiple=False,
                file_types=("PyTorch checkpoint (*.pth)", "All files (*.*)"),
            )
        except Exception as exc:
            print(f"[studio] open dialog failed: {exc}", flush=True)
            return None
        if not result:
            return None
        chosen = result[0] if isinstance(result, (list, tuple)) else result
        return str(chosen) if chosen else None

    return import_via_dialog


def _run_window(webview: Any, server_mod: Any, url: str) -> None:
    webview.create_window(
        "WaveDiT Studio",
        url=url,
        width=1480,
        height=940,
        min_size=(1120, 760),
        background_color="#0b0d12",
    )
    server_mod.export_handler = _make_export_handler(webview)
    server_mod.import_handler = _make_import_handler(webview)
    # pywebview >= 6 installs a native app menu with a working Edit menu (clipboard
    # shortcuts in WKWebView text fields) on its own; no custom menu needed.
    webview.start()
    # webview.start() returns when the window closes; the server thread is a
    # daemon, so falling out of main() ends the process.
    print("[studio] window closed, exiting.", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.selfcheck:
        _selfcheck()
    if args.port is not None:
        os.environ["WAVEDIT_STUDIO_PORT"] = str(args.port)

    from studio import server

    _httpd, port, _hub = server.serve_in_thread()
    url = f"http://127.0.0.1:{port}/"

    webview_mod: Any = None
    if os.environ.get("WAVEDIT_STUDIO_HEADLESS") != "1":
        try:
            import webview as webview_mod  # type: ignore[no-redef]
        except Exception as exc:
            print(f"[studio] pywebview unavailable ({exc}); falling back to headless.",
                  flush=True)
            webview_mod = None

    if webview_mod is None:
        _run_headless(url)
    else:
        _run_window(webview_mod, server, url)


if __name__ == "__main__":
    main()

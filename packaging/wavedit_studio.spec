# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for WaveDiT Studio: a windowed macOS .app bundle (arm64).

Entry-point design
------------------
PyInstaller analyzes studio/main.py as the entry *script*: in the frozen app it
runs as __main__, exactly like `python studio/main.py` in development. The rest of
the `studio` package and the vendored `wavedit` package are bundled as importable
modules through explicit hiddenimports plus pathex:

  * pathex includes the repo root -> `import studio.<mod>` resolves
  * pathex includes studio/        -> `import wavedit.<mod>` resolves, because the
    vendored tree uses absolute top-level imports ("from wavedit ...")

main.py already resolves the UI directory through sys._MEIPASS/studio/ui when
sys.frozen is set, so the only data we must carry is the studio/ui tree (the
Python modules travel inside the PYZ archive; duplicating the whole studio
source tree as data is unnecessary and would only inflate the bundle).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()   # noqa: F821 - SPECPATH is injected by PyInstaller
ROOT_DIR = SPEC_DIR.parent            # repo root (packaging/ lives directly under it)
MACOS_DIR = ROOT_DIR                  # build root for icon/dist outputs
STUDIO_DIR = ROOT_DIR / "studio"

# Single source of truth for the version (same extraction build.sh performs).
_version_match = re.search(
    r'__version__\s*=\s*"([^"]+)"', (STUDIO_DIR / "__init__.py").read_text()
)
VERSION = _version_match.group(1) if _version_match else "1.0.0"

# collect_submodules() imports the target package in an isolated child
# interpreter that sees only the environment, not this process's sys.path.
# `studio` and the vendored `wavedit` are plain source trees (not installed
# wheels), so expose them via PYTHONPATH for those child imports to succeed.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(MACOS_DIR), str(STUDIO_DIR)]
    + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)

hiddenimports: list[str] = []

# Vendored model code: every submodule named explicitly so static analysis
# cannot miss config-driven or lazily imported pieces.
hiddenimports += collect_submodules("wavedit")

# The studio package itself. studio.wavedit (the same files seen as a
# subpackage of studio) is filtered out: it is already bundled as top-level
# `wavedit`, and double-bundling it would only add dead weight.
hiddenimports += [
    m for m in collect_submodules("studio") if not m.startswith("studio.wavedit")
]

for pkg in ("einops", "huggingface_hub", "yaml", "webview"):
    hiddenimports += collect_submodules(pkg)

# Non-Python assets the app reads at runtime: the web UI (index.html, app.css,
# app.js, vendored niivue.umd.js) lands at <bundle>/studio/ui.
datas: list[tuple[str, str]] = [(str(STUDIO_DIR / "ui"), "studio/ui")]
binaries: list[tuple[str, str]] = []

# Heavyweight packages with compiled extensions and data files: take everything.
# pyinstaller-hooks-contrib covers most of this already; collect_all is belt and
# suspenders so MPS dylibs and package metadata are never silently dropped.
for pkg in ("torch", "nibabel", "pywt"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "PIL",
    "wandb",
    "pytest",
    "tests",
    "test",
]

a = Analysis(
    [str(STUDIO_DIR / "main.py")],
    pathex=[str(MACOS_DIR), str(STUDIO_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WaveDiT Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # windowed app, no terminal
    target_arch="arm64",
    codesign_identity=None,  # build.sh ad-hoc signs the finished bundle
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="WaveDiT Studio",
)

app = BUNDLE(
    coll,
    name="WaveDiT Studio.app",
    icon=str(MACOS_DIR / "build" / "WaveDiT.icns"),
    bundle_identifier="it.poliba.sisinflab.wavedit-studio",
    version=VERSION,
    info_plist={
        "CFBundleName": "WaveDiT Studio",
        "CFBundleDisplayName": "WaveDiT Studio",
        "CFBundleShortVersionString": VERSION,
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": (
            "Copyright SisInfLab, Politecnico di Bari. "
            "Research artifact: synthetic images only, not a medical device."
        ),
        "LSApplicationCategoryType": "public.app-category.medical",
    },
)

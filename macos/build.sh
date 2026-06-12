#!/usr/bin/env bash
# WaveDiT Studio: one-command build on an Apple Silicon Mac -> dist/WaveDiT-Studio-<version>.dmg
#
# Only stock macOS tools are required (codesign, iconutil, hdiutil, sips, osascript).
# No Xcode command line tools, no Homebrew. Network is needed for pip wheels and,
# if the vendored viewer is missing, one CDN fetch.
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$MACOS_DIR"

banner() { printf '\n========== %s ==========\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

APP_NAME="WaveDiT Studio"
VENV="$MACOS_DIR/.venv-build"
NIIVUE_REL="studio/ui/vendor/niivue.umd.js"
NIIVUE_SHA256_B64="R7iWt37Epb4+8ZScM605O29FYpkh8ybCeQALz1G9tK8="
NIIVUE_URL="https://cdn.jsdelivr.net/npm/@niivue/niivue@0.69.0/dist/niivue.umd.js"

# App version: read from studio/__init__.py, fall back to 1.0.0.
VERSION="$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' studio/__init__.py 2>/dev/null | head -n1 || true)"
VERSION="${VERSION:-1.0.0}"

banner "WaveDiT Studio build $VERSION"

# --- Step 1: platform guards -------------------------------------------------
banner "Step 1/8: platform checks"
[[ "$(uname -s)" == "Darwin" ]] \
  || die "this build script runs on macOS only (got: $(uname -s)). See README.md for the cross-platform dev mode."
[[ "$(uname -m)" == "arm64" ]] \
  || die "Apple Silicon (arm64) is required (got: $(uname -m)). Intel Macs are not supported."
MACOS_VER="$(sw_vers -productVersion)"
MACOS_MAJOR="${MACOS_VER%%.*}"
[[ "$MACOS_MAJOR" -ge 13 ]] || die "macOS 13 (Ventura) or newer is required (got: $MACOS_VER)."
for tool in codesign iconutil hdiutil sips osascript; do
  command -v "$tool" >/dev/null 2>&1 || die "missing stock macOS tool: $tool"
done
echo "macOS $MACOS_VER on arm64: OK (Xcode command line tools are NOT required)"

# --- Step 2: Python build environment ---------------------------------------
banner "Step 2/8: Python build environment"
PIP=()
if command -v uv >/dev/null 2>&1; then
  echo "using uv: $(command -v uv)"
  if ! uv python find 3.12 >/dev/null 2>&1; then
    echo "uv found no CPython 3.12 on this machine; it can download a managed build."
    read -r -p "Allow uv to download CPython 3.12? [y/N] " REPLY
    [[ "$REPLY" == "y" || "$REPLY" == "Y" ]] \
      || die "declined. Install Python 3.11+ (python.org) or let uv manage one, then re-run."
  fi
  uv venv --python 3.12 "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
elif command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)'; then
  echo "using system python3: $(python3 --version) at $(command -v python3)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  PIP=("$VENV/bin/pip" install)
else
  cat >&2 <<'EOF'
ERROR: no suitable Python found. This script never installs anything on its own.
Install one of the following, then re-run ./build.sh:
  * uv (recommended):   brew install uv
                        or: curl -LsSf https://astral.sh/uv/install.sh | sh
  * CPython 3.11-3.13:  https://www.python.org/downloads/macos/
                        (torch 2.8.0 has no wheels for 3.14+)
EOF
  exit 1
fi
"${PIP[@]}" -r requirements.txt 'pyinstaller==6.*'
PYBIN="$VENV/bin/python"
echo "build venv ready: $("$PYBIN" -V) at $VENV"

# --- Step 3: verify the vendored Niivue viewer -------------------------------
banner "Step 3/8: verify vendored Niivue viewer"
niivue_hash() { openssl dgst -sha256 -binary "$1" | base64; }
if [[ ! -f "$NIIVUE_REL" ]]; then
  echo "vendored viewer missing, fetching the pinned 0.69.0 build from jsdelivr"
  mkdir -p "$(dirname "$NIIVUE_REL")"
  curl -fL --retry 3 -o "$NIIVUE_REL" "$NIIVUE_URL"
fi
ACTUAL_HASH="$(niivue_hash "$NIIVUE_REL")"
if [[ "$ACTUAL_HASH" != "$NIIVUE_SHA256_B64" ]]; then
  die "niivue.umd.js sha256 mismatch
  expected: $NIIVUE_SHA256_B64
  actual:   $ACTUAL_HASH
Delete $NIIVUE_REL and re-run (it will be re-fetched), or restore it from git."
fi
echo "niivue.umd.js sha256: OK"

# --- Step 4: icon + DMG background -------------------------------------------
banner "Step 4/8: render app icon and DMG background"
"$PYBIN" packaging/make_icon.py --out build/icon
iconutil -c icns build/icon/WaveDiT.iconset -o build/WaveDiT.icns
echo "wrote build/WaveDiT.icns"

# --- Step 5: freeze the app with PyInstaller ---------------------------------
banner "Step 5/8: PyInstaller freeze"
"$VENV/bin/pyinstaller" --noconfirm packaging/wavedit_studio.spec
[[ -d "dist/$APP_NAME.app" ]] || die "PyInstaller did not produce dist/$APP_NAME.app"

# --- Step 6: smoke test -------------------------------------------------------
banner "Step 6/8: smoke test (--selfcheck)"
"dist/$APP_NAME.app/Contents/MacOS/$APP_NAME" --selfcheck \
  || die "the bundled app failed its self check"

# --- Step 7: ad-hoc code signing ----------------------------------------------
banner "Step 7/8: ad-hoc codesign"
codesign --force --deep -s - "dist/$APP_NAME.app"
codesign -dv "dist/$APP_NAME.app"

# --- Step 8: package the DMG ----------------------------------------------------
banner "Step 8/8: package DMG"
bash packaging/make_dmg.sh "dist/$APP_NAME.app" "$VERSION"
DMG="dist/WaveDiT-Studio-$VERSION.dmg"
[[ -f "$DMG" ]] || die "DMG was not produced at $DMG"

# --- Summary -------------------------------------------------------------------
banner "Build complete"
DMG_SIZE="$(du -h "$DMG" | cut -f1 | tr -d '[:space:]')"
cat <<EOF
  DMG: $MACOS_DIR/$DMG ($DMG_SIZE)

Install:
  1. Open the DMG and drag "WaveDiT Studio" into Applications.
  2. First launch only: the app is ad-hoc signed, so Gatekeeper warns once.
     Right-click (control-click) the app, choose "Open", then confirm.
     Alternative: xattr -dr com.apple.quarantine "/Applications/WaveDiT Studio.app"
  3. On first run the app offers to download model weights from
     huggingface.co/danesed/WaveDiT.

WaveDiT Studio generates synthetic research images. It is not a medical device.
EOF

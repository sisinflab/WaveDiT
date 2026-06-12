#!/usr/bin/env bash
# Package a built .app into a distributable DMG with a drag-to-Applications layout.
#
# Usage: make_dmg.sh <path-to-app> [version]
#
# Pipeline: staging dir (.app via ditto + /Applications symlink + background)
# -> hdiutil UDRW image -> mount -> Finder layout via osascript -> detach
# -> hdiutil convert UDZO -> dist/WaveDiT-Studio-<version>.dmg
#
# Every osascript step is best-effort: when Finder scripting is unavailable
# (SSH session, CI, automation permissions denied) a plain but fully functional
# DMG is still produced.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_DIR="$(dirname "$SCRIPT_DIR")"

APP_PATH="${1:?usage: make_dmg.sh <path-to-app> [version]}"
VERSION="${2:-1.0.0}"
[[ -d "$APP_PATH" ]] || { echo "ERROR: app bundle not found: $APP_PATH" >&2; exit 1; }

APP_NAME="$(basename "$APP_PATH" .app)"
VOL_NAME="$APP_NAME"
STAGING="$MACOS_DIR/build/dmg-staging"
RW_DMG="$MACOS_DIR/build/${APP_NAME// /-}-rw.dmg"
OUT_DMG="$MACOS_DIR/dist/WaveDiT-Studio-$VERSION.dmg"
BACKGROUND="$MACOS_DIR/build/icon/dmg_background.png"
MOUNT_POINT=""

notice() { printf 'notice: %s\n' "$*"; }

cleanup() {
  if [[ -n "$MOUNT_POINT" && -d "$MOUNT_POINT" ]]; then
    hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
  fi
  rm -f "$RW_DMG"
}
trap cleanup EXIT

echo "staging $APP_NAME ..."
rm -rf "$STAGING"
mkdir -p "$STAGING/.background" "$MACOS_DIR/dist"
ditto "$APP_PATH" "$STAGING/$APP_NAME.app"
ln -s /Applications "$STAGING/Applications"
if [[ -f "$BACKGROUND" ]]; then
  cp "$BACKGROUND" "$STAGING/.background/dmg_background.png"
else
  notice "background image not found at $BACKGROUND; the DMG will use a plain window"
fi

echo "creating writable image ..."
rm -f "$RW_DMG"
hdiutil create -srcfolder "$STAGING" -volname "$VOL_NAME" -fs HFS+ \
  -format UDRW -ov "$RW_DMG" >/dev/null

echo "mounting for Finder layout ..."
ATTACH_OUT="$(hdiutil attach -readwrite -noverify -noautoopen "$RW_DMG")"
MOUNT_POINT="$(printf '%s\n' "$ATTACH_OUT" | grep -o '/Volumes/.*' | tail -n 1)"
# A pre-existing "WaveDiT Studio" volume would make Finder mount this one at
# "WaveDiT Studio 1": always address the disk by its actual mount-point name.
VOL_NAME="$(basename "$MOUNT_POINT")"

if [[ -n "$MOUNT_POINT" && -d "$MOUNT_POINT" ]]; then
  # Window: 680x440 at a sane screen position; large icons; no chrome.
  osascript >/dev/null <<EOF || notice "Finder window styling skipped (scripting unavailable)"
tell application "Finder"
  tell disk "$VOL_NAME"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {200, 120, 880, 560}
    set viewOptions to the icon view options of container window
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 96
  end tell
end tell
EOF
  if [[ -f "$STAGING/.background/dmg_background.png" ]]; then
    osascript >/dev/null <<EOF || notice "background image assignment skipped"
tell application "Finder"
  tell disk "$VOL_NAME"
    set viewOptions to the icon view options of container window
    set background picture of viewOptions to file ".background:dmg_background.png"
  end tell
end tell
EOF
  fi
  osascript >/dev/null <<EOF || notice "icon positioning skipped"
tell application "Finder"
  tell disk "$VOL_NAME"
    set position of item "$APP_NAME.app" of container window to {165, 210}
    set position of item "Applications" of container window to {515, 210}
    close
    open
    update without registering applications
    close
  end tell
end tell
EOF
  sync || true
fi

echo "detaching ..."
DETACHED=0
for attempt in 1 2 3 4 5; do
  if [[ "$attempt" -lt 5 ]]; then
    hdiutil detach "$MOUNT_POINT" >/dev/null 2>&1 && DETACHED=1 && break
    sleep 2
  else
    # Last attempt: force, and let its stderr through so a failure is visible.
    hdiutil detach "$MOUNT_POINT" -force && DETACHED=1
  fi
done
if [[ "$DETACHED" -ne 1 || -d "$MOUNT_POINT" ]]; then
  echo "ERROR: could not detach $MOUNT_POINT; close any Finder window using it and re-run." >&2
  exit 1
fi
MOUNT_POINT=""

echo "compressing to UDZO ..."
rm -f "$OUT_DMG"
hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$OUT_DMG" >/dev/null
echo "wrote $OUT_DMG"

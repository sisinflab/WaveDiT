#!/usr/bin/env python3
"""Render the WaveDiT Studio app icon set and the DMG background.

Pure numpy + stdlib (zlib, struct), no Pillow. The icon is rendered analytically
at 4096x4096 (4x supersampling), box-downsampled to a 1024 master, then
area-averaged down to every size iconutil expects. A 660x420 DMG background is
rendered alongside it.

Usage:  python make_icon.py --out build/icon
Output: <out>/WaveDiT.iconset/icon_*.png  and  <out>/dmg_background.png
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

BASE = 1024     # master icon size
SS = 4          # supersampling factor (render at BASE * SS)

# (file name per the iconutil naming rules, pixel size)
ICONSET_FILES: list[tuple[str, int]] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

GRAD_TOP_LEFT = "#6366f1"      # indigo
GRAD_BOTTOM_RIGHT = "#a855f7"  # violet
DMG_BG = "#0b0d12"             # near-black


# ----------------------------------------------------------------------------- PNG

def encode_rgba_png(arr: np.ndarray) -> bytes:
    """Encode an (H, W, 4) uint8 array as an 8-bit RGBA PNG (filter 0 rows)."""
    if arr.ndim != 3 or arr.shape[2] != 4 or arr.dtype != np.uint8:
        raise ValueError("expected an (H, W, 4) uint8 array")
    h, w = arr.shape[:2]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit, color type 6 (RGBA)
    rows = np.zeros((h, 1 + w * 4), dtype=np.uint8)      # leading 0: per-row filter byte
    rows[:, 1:] = arr.reshape(h, w * 4)
    idat = zlib.compress(rows.tobytes(), 9)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- helpers

def hex_rgb(code: str) -> np.ndarray:
    """'#rrggbb' -> float32 RGB in [0, 1]."""
    return np.array([int(code[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255.0


def box_downsample(pm: np.ndarray, size: int) -> np.ndarray:
    """Area-average a square premultiplied RGBA float image down to size x size."""
    k = pm.shape[0] // size
    if k == 1:
        return pm
    return pm.reshape(size, k, size, k, pm.shape[2]).mean(axis=(1, 3))


def premul_to_uint8(pm: np.ndarray) -> np.ndarray:
    """Premultiplied float RGBA -> straight-alpha uint8 RGBA."""
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([rgb, a], axis=-1)
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def wavelet_curve(u: np.ndarray, f: float, amp: float, phi: float,
                  c: float, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Morlet-style stroke y(u) = amp * cos(2 pi f u + phi) * gaussian(u; c, s).

    Returns (y, dy/du, envelope) on the normalized abscissa u in [0, 1].
    """
    env = np.exp(-((u - c) ** 2) / (2.0 * s * s))
    phase = 2.0 * np.pi * f * u + phi
    y = amp * np.cos(phase) * env
    denv = env * (-(u - c) / (s * s))
    dy = amp * (-2.0 * np.pi * f * np.sin(phase) * env + np.cos(phase) * denv)
    return y, dy, env


def composite_stroke(rgb: np.ndarray, ys: np.ndarray, ycurve: np.ndarray,
                     slope: np.ndarray, env: np.ndarray, width: float,
                     alpha: float, color: np.ndarray, aa: float) -> np.ndarray:
    """Anti-aliased stroke + soft glow composited over an RGB image.

    Distance to the curve is approximated by the vertical distance corrected by
    the local slope, which is accurate for the moderate slopes used here.
    """
    dist = np.abs(ys - ycurve[None, :]) / np.sqrt(1.0 + slope[None, :] ** 2)
    core = np.clip((width / 2.0 + aa - dist) / aa, 0.0, 1.0)
    sigma = width * 1.6
    glow = 0.35 * np.exp(-(dist * dist) / (2.0 * sigma * sigma))
    a = np.clip(core + glow, 0.0, 1.0) * alpha * (env[None, :] ** 0.75)
    return rgb * (1.0 - a[..., None]) + color[None, None, :] * a[..., None]


# ------------------------------------------------------------------------------ icon

def render_icon_master() -> np.ndarray:
    """Render the 1024x1024 premultiplied RGBA float master icon."""
    n = BASE * SS
    xs = np.arange(n, dtype=np.float32)[None, :]
    ys = np.arange(n, dtype=np.float32)[:, None]
    cx = cy = (n - 1) / 2.0

    # Big Sur style rounded square: 824/1024 content grid, radius 22.5% of the
    # content size. A Minkowski exponent slightly above 2 approximates the
    # continuous-corner (squircle) profile of native macOS icons.
    content = 824.0 * SS
    half = content / 2.0
    radius = 0.225 * content
    p = 2.6
    qx = np.maximum(np.abs(xs - cx) - (half - radius), 0.0)
    qy = np.maximum(np.abs(ys - cy) - (half - radius), 0.0)
    sdf = (qx ** p + qy ** p) ** (1.0 / p) - radius
    aa = 1.5 * SS
    mask = np.clip(0.5 - sdf / aa, 0.0, 1.0)

    # Diagonal indigo -> violet gradient with a subtle vertical luminosity falloff.
    t = (xs + ys) / (2.0 * (n - 1))
    c0, c1 = hex_rgb(GRAD_TOP_LEFT), hex_rgb(GRAD_BOTTOM_RIGHT)
    rgb = c0[None, None, :] * (1.0 - t[..., None]) + c1[None, None, :] * t[..., None]
    lum = (1.06 - 0.14 * (ys / (n - 1)))
    rgb = np.clip(rgb * lum[..., None], 0.0, 1.0)

    # Three white wavelet strokes (the brand mark): one bold opaque line and two
    # thinner translucent echoes with different frequency, amplitude and offset.
    u = (xs[0] - (cx - half)) / content
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    strokes = [
        # f,   amp(px@1024), phi,  c,    s,    width(px@1024), alpha, dy(px@1024)
        (3.0, 150.0, 0.00, 0.50, 0.165, 14.0, 1.00, 0.0),
        (4.2, 105.0, 1.30, 0.46, 0.140, 8.0, 0.45, -52.0),
        (2.2, 175.0, -0.90, 0.55, 0.205, 8.0, 0.30, 58.0),
    ]
    for f, amp, phi, c, s, w, alpha, dy in strokes:
        ycv, dydu, env = wavelet_curve(u, f, amp * SS, phi, c, s)
        slope = dydu / content
        rgb = composite_stroke(rgb, ys, cy + dy * SS + ycv, slope, env,
                               w * SS, alpha, white, aa)

    pm = np.empty((n, n, 4), dtype=np.float32)
    pm[..., :3] = rgb * mask[..., None]
    pm[..., 3] = mask
    return box_downsample(pm, BASE)


# ----------------------------------------------------------------------- pixel font

# Minimal 5x7 glyph set covering the DMG background text only.
FONT: dict[str, list[str]] = {
    " ": ["....."] * 7,
    "A": [".XXX.", "X...X", "X...X", "XXXXX", "X...X", "X...X", "X...X"],
    "D": ["XXXX.", "X...X", "X...X", "X...X", "X...X", "X...X", "XXXX."],
    "S": [".XXXX", "X....", "X....", ".XXX.", "....X", "....X", "XXXX."],
    "T": ["XXXXX", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."],
    "W": ["X...X", "X...X", "X...X", "X.X.X", "X.X.X", "X.X.X", ".X.X."],
    "a": [".....", ".....", ".XXX.", "....X", ".XXXX", "X...X", ".XXXX"],
    "c": [".....", ".....", ".XXX.", "X....", "X....", "X....", ".XXX."],
    "d": ["....X", "....X", ".XXXX", "X...X", "X...X", "X...X", ".XXXX"],
    "e": [".....", ".....", ".XXX.", "X...X", "XXXXX", "X....", ".XXX."],
    "g": [".....", ".....", ".XXXX", "X...X", ".XXXX", "....X", ".XXX."],
    "i": ["..X..", ".....", "..X..", "..X..", "..X..", "..X..", "..X.."],
    "l": ["..X..", "..X..", "..X..", "..X..", "..X..", "..X..", "..X.."],
    "n": [".....", ".....", "X.XX.", "XX..X", "X...X", "X...X", "X...X"],
    "o": [".....", ".....", ".XXX.", "X...X", "X...X", "X...X", ".XXX."],
    "p": [".....", ".....", "XXXX.", "X...X", "XXXX.", "X....", "X...."],
    "r": [".....", ".....", "X.XX.", "XX...", "X....", "X....", "X...."],
    "s": [".....", ".....", ".XXXX", "X....", ".XXX.", "....X", "XXXX."],
    "t": ["..X..", "..X..", "XXXXX", "..X..", "..X..", "..X..", "...XX"],
    "u": [".....", ".....", "X...X", "X...X", "X...X", "X...X", ".XXXX"],
    "v": [".....", ".....", "X...X", "X...X", "X...X", ".X.X.", "..X.."],
}


def text_width(text: str, scale: int) -> int:
    return (6 * len(text) - 1) * scale


def draw_text(alpha: np.ndarray, text: str, x: int, y: int,
              scale: int, value: float) -> None:
    """Stamp text into an alpha map using the 5x7 pixel font (6px advance)."""
    pen = x
    for ch in text:
        rows = FONT.get(ch, FONT[" "])
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "X":
                    alpha[y + r * scale: y + (r + 1) * scale,
                          pen + c * scale: pen + (c + 1) * scale] = value
        pen += 6 * scale


# -------------------------------------------------------------------- DMG background

def render_dmg_background() -> np.ndarray:
    """660x420 RGBA background: faint wave mark, title, drag arrow and caption."""
    w, h = 660, 420
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    rgb = np.broadcast_to(hex_rgb(DMG_BG)[None, None, :], (h, w, 3)).copy()

    # Faint brand mark at the center top, tinted with the icon gradient.
    mark_cx, mark_cy, mark_w = 330.0, 56.0, 280.0
    u = (xs[0] - (mark_cx - mark_w / 2.0)) / mark_w
    grad_t = np.clip(u, 0.0, 1.0)
    c0, c1 = hex_rgb(GRAD_TOP_LEFT), hex_rgb(GRAD_BOTTOM_RIGHT)
    mark_color_row = c0[None, :] * (1.0 - grad_t[:, None]) + c1[None, :] * grad_t[:, None]
    aa = 1.25
    for f, amp, phi, c, s, width, alpha in [
        (3.0, 22.0, 0.00, 0.50, 0.165, 3.0, 0.85),
        (4.2, 16.0, 1.30, 0.46, 0.140, 1.8, 0.40),
    ]:
        ycv, dydu, env = wavelet_curve(u, f, amp, phi, c, s)
        slope = dydu / mark_w
        dist = np.abs(ys - (mark_cy + ycv)[None, :]) / np.sqrt(1.0 + slope[None, :] ** 2)
        core = np.clip((width / 2.0 + aa - dist) / aa, 0.0, 1.0)
        glow = 0.30 * np.exp(-(dist * dist) / (2.0 * (width * 2.0) ** 2))
        a = np.clip(core + glow, 0.0, 1.0) * alpha * (env[None, :] ** 0.75)
        rgb = rgb * (1.0 - a[..., None]) + mark_color_row[None, :, :] * a[..., None]

    # Text layers (crisp pixel font, no anti-aliasing needed at these scales).
    text_a = np.zeros((h, w), dtype=np.float32)
    title = "WaveDiT Studio"
    draw_text(text_a, title, int(330 - text_width(title, 3) / 2), 102, 3, 0.92)
    caption = "drag WaveDiT Studio into Applications"
    draw_text(text_a, caption, int(330 - text_width(caption, 2) / 2), 326, 2, 0.55)

    # Arrow from the app position (165, 210) toward Applications (515, 210).
    arrow_a = np.zeros((h, w), dtype=np.float32)
    y0, x0, x1, tip = 210.0, 244.0, 408.0, 436.0
    shaft = (np.clip(3.5 - np.abs(ys - y0), 0.0, 1.0)
             * np.clip(xs - x0 + 1.0, 0.0, 1.0) * np.clip(x1 - xs + 1.0, 0.0, 1.0))
    head_half = np.clip((tip - xs) * 0.5, 0.0, 14.0)
    head = (np.clip(head_half - np.abs(ys - y0) + 0.5, 0.0, 1.0)
            * np.clip(xs - x1 + 1.0, 0.0, 1.0) * np.clip(tip - xs + 1.0, 0.0, 1.0))
    arrow_a = np.clip(shaft + head, 0.0, 1.0) * 0.55

    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for a in (text_a, arrow_a):
        rgb = rgb * (1.0 - a[..., None]) + white[None, None, :] * a[..., None]

    out = np.empty((h, w, 4), dtype=np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    out[..., 3] = 1.0
    return (out * 255.0 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="build/icon",
                        help="output directory (default: build/icon)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    iconset = out_dir / "WaveDiT.iconset"
    iconset.mkdir(parents=True, exist_ok=True)

    master = render_icon_master()
    for name, size in ICONSET_FILES:
        img = premul_to_uint8(box_downsample(master, size))
        (iconset / name).write_bytes(encode_rgba_png(img))
        print(f"wrote {iconset / name} ({size}x{size})")

    bg_path = out_dir / "dmg_background.png"
    bg_path.write_bytes(encode_rgba_png(render_dmg_background()))
    print(f"wrote {bg_path} (660x420)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

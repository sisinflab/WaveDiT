"""Minimal 8-bit grayscale PNG encoder using only zlib and struct (no Pillow)."""

from __future__ import annotations

import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """Build one PNG chunk: length, tag, payload, CRC32 over tag + payload."""
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_gray_png(arr: np.ndarray) -> bytes:
    """Encode a 2D uint8 array as an 8-bit grayscale PNG (filter type 0 on every row)."""
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected dtype uint8, got {arr.dtype}")
    height, width = arr.shape
    if height < 1 or width < 1:
        raise ValueError(f"image dimensions must be positive, got {arr.shape}")
    # IHDR: width, height, bit depth 8, color type 0 (grayscale),
    # compression 0, filter 0, interlace 0.
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    # Scanlines: one leading filter byte (0 = None) per row, then raw pixels.
    scanlines = np.empty((height, width + 1), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = arr
    idat = zlib.compress(scanlines.tobytes(), 9)
    return _SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")

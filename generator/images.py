"""Tile uint8 images into a grid and write greyscale PNGs, using only numpy and the standard library."""

import math
import struct
import zlib
from pathlib import Path

import numpy as np


def make_grid(images: np.ndarray, columns: int | None = None, padding: int = 2) -> np.ndarray:
    """Tile uint8 images (N, H, W) into one (rows * (H + p) + p, columns * (W + p) + p) image on black."""
    n, h, w = images.shape
    columns = columns or math.ceil(math.sqrt(n))
    rows = math.ceil(n / columns)
    grid = np.zeros((rows * (h + padding) + padding, columns * (w + padding) + padding), dtype=np.uint8)
    for i, image in enumerate(images):
        top, left = padding + (i // columns) * (h + padding), padding + (i % columns) * (w + padding)
        grid[top : top + h, left : left + w] = image
    return grid


def write_png(path: Path, image: np.ndarray):
    """Write a uint8 (H, W) greyscale image as a PNG using only the standard library."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    height, width = image.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(image, dtype=np.uint8))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit greyscale, no interlacing
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    )

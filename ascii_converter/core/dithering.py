"""
dithering.py
------------
Error-diffusion and ordered dithering for luminance maps.

Supported algorithms
--------------------
  floyd_steinberg  — classic, fastest error-diffusion
  stucki           — wider kernel, slightly smoother
  jarvis           — Jarvis-Judice-Ninke, wide kernel
  bayer            — ordered 8×8 Bayer matrix
  random           — additive random noise quantisation
  none             — no dithering (plain nearest-level quantisation)

All functions operate on float32 arrays in [0, 1] and return float32 in [0, 1].
"""

from __future__ import annotations

import numpy as np
from typing import Literal

DitherMethod = Literal[
    "floyd_steinberg", "stucki", "jarvis", "bayer", "random", "none"
]

# ---------------------------------------------------------------------------
# Bayer 8×8 threshold matrix (normalised to [0, 1])
# ---------------------------------------------------------------------------
_BAYER_8 = np.array([
    [ 0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [ 3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) / 64.0  # normalise to [0, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_dithering(
    luma: np.ndarray,
    method: DitherMethod = "floyd_steinberg",
    levels: int = 16,
) -> np.ndarray:
    """
    Apply dithering to a float32 luminance map in [0, 1].

    Parameters
    ----------
    luma   : (H, W) float32
    method : dithering algorithm name
    levels : number of quantisation levels (e.g. len(charset))

    Returns
    -------
    (H, W) float32 dithered luminance, same value range.
    """
    luma = luma.astype(np.float32).copy()

    if method == "none":
        return luma
    elif method == "bayer":
        return _bayer(luma, levels)
    elif method == "random":
        return _random_dither(luma, levels)
    elif method == "floyd_steinberg":
        return _error_diffusion(luma, levels, _FS_KERNEL)
    elif method == "stucki":
        return _error_diffusion(luma, levels, _STUCKI_KERNEL)
    elif method == "jarvis":
        return _error_diffusion(luma, levels, _JARVIS_KERNEL)
    else:
        raise ValueError(f"Unknown dithering method: {method!r}")


# ---------------------------------------------------------------------------
# Error-diffusion kernels
# Each kernel is a list of (row_offset, col_offset, weight) tuples.
# ---------------------------------------------------------------------------

_FS_KERNEL = [
    (0,  1, 7 / 16),
    (1, -1, 3 / 16),
    (1,  0, 5 / 16),
    (1,  1, 1 / 16),
]

_STUCKI_KERNEL = [
    (0,  1, 8 / 42),
    (0,  2, 4 / 42),
    (1, -2, 2 / 42),
    (1, -1, 4 / 42),
    (1,  0, 8 / 42),
    (1,  1, 4 / 42),
    (1,  2, 2 / 42),
    (2, -2, 1 / 42),
    (2, -1, 2 / 42),
    (2,  0, 4 / 42),
    (2,  1, 2 / 42),
    (2,  2, 1 / 42),
]

_JARVIS_KERNEL = [
    (0,  1, 7 / 48),
    (0,  2, 5 / 48),
    (1, -2, 3 / 48),
    (1, -1, 5 / 48),
    (1,  0, 7 / 48),
    (1,  1, 5 / 48),
    (1,  2, 3 / 48),
    (2, -2, 1 / 48),
    (2, -1, 3 / 48),
    (2,  0, 5 / 48),
    (2,  1, 3 / 48),
    (2,  2, 1 / 48),
]


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def _quantise(value: float, levels: int) -> float:
    """Round value to nearest quantisation level."""
    step = 1.0 / (levels - 1)
    return round(value / step) * step


def _error_diffusion(
    luma: np.ndarray,
    levels: int,
    kernel: list,
) -> np.ndarray:
    """
    Generic error-diffusion dithering.
    Operates in-place for speed; returns the modified array.
    Pure-Python inner loop is unavoidable for sequential error propagation,
    but the kernel is small so it stays fast in practice.
    """
    H, W = luma.shape
    step = 1.0 / (levels - 1)

    for y in range(H):
        for x in range(W):
            old = luma[y, x]
            new = round(old / step) * step
            new = max(0.0, min(1.0, new))
            luma[y, x] = new
            err = old - new
            for dy, dx, w in kernel:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    luma[ny, nx] += err * w

    return luma


def _bayer(luma: np.ndarray, levels: int) -> np.ndarray:
    """Ordered Bayer dithering — fully vectorised."""
    H, W = luma.shape
    # Tile Bayer matrix to image size
    tile_rows = (H + 7) // 8
    tile_cols = (W + 7) // 8
    bayer = np.tile(_BAYER_8, (tile_rows, tile_cols))[:H, :W]

    step = 1.0 / (levels - 1)
    # Add sub-pixel offset before quantising
    offset = bayer * step - step / 2.0
    dithered = luma + offset
    dithered = np.clip(dithered, 0.0, 1.0)
    # Quantise
    dithered = np.round(dithered / step) * step
    return np.clip(dithered, 0.0, 1.0)


def _random_dither(luma: np.ndarray, levels: int) -> np.ndarray:
    """Additive random noise before quantisation."""
    step = 1.0 / (levels - 1)
    noise = (np.random.random(luma.shape).astype(np.float32) - 0.5) * step
    dithered = np.clip(luma + noise, 0.0, 1.0)
    dithered = np.round(dithered / step) * step
    return np.clip(dithered, 0.0, 1.0)

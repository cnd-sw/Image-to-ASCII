"""
colorizer.py
------------
Converts a per-cell BGR colour array into the appropriate ANSI escape codes
or other colour representations.

Supported modes
---------------
  none       — no colour codes; plain text only
  ansi16     — 16-colour ANSI (closest standard colour)
  ansi256    — 256-colour xterm palette
  truecolor  — 24-bit \x1b[38;2;r;g;bm  (most modern terminals)
  html       — inline CSS colour for HTML output
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

# ---------------------------------------------------------------------------
# 16-colour ANSI palette  (r, g, b)
# ---------------------------------------------------------------------------
_ANSI16_PALETTE = np.array([
    [  0,   0,   0],   # 0  black
    [170,   0,   0],   # 1  red
    [  0, 170,   0],   # 2  green
    [170, 170,   0],   # 3  yellow/dark
    [  0,   0, 170],   # 4  blue
    [170,   0, 170],   # 5  magenta
    [  0, 170, 170],   # 6  cyan
    [170, 170, 170],   # 7  light grey
    [ 85,  85,  85],   # 8  dark grey (bright black)
    [255,  85,  85],   # 9  bright red
    [ 85, 255,  85],   # 10 bright green
    [255, 255,  85],   # 11 bright yellow
    [ 85,  85, 255],   # 12 bright blue
    [255,  85, 255],   # 13 bright magenta
    [ 85, 255, 255],   # 14 bright cyan
    [255, 255, 255],   # 15 white
], dtype=np.float32)

# ---------------------------------------------------------------------------
# 256-colour xterm palette helpers
# ---------------------------------------------------------------------------

def _build_xterm256() -> np.ndarray:
    """Return (256, 3) float32 array of the xterm-256 palette in RGB."""
    palette = np.zeros((256, 3), dtype=np.float32)
    # First 16: standard ANSI colours
    palette[:16] = _ANSI16_PALETTE
    # 216 colour cube (6×6×6) starting at index 16
    for i in range(216):
        b = i % 6
        g = (i // 6) % 6
        r = i // 36
        palette[16 + i] = [
            0 if r == 0 else 55 + 40 * r,
            0 if g == 0 else 55 + 40 * g,
            0 if b == 0 else 55 + 40 * b,
        ]
    # 24 greyscale ramp starting at index 232
    for i in range(24):
        v = 8 + 10 * i
        palette[232 + i] = [v, v, v]
    return palette

_XTERM256 = _build_xterm256()


# ---------------------------------------------------------------------------
# Core Colorizer class
# ---------------------------------------------------------------------------

class Colorizer:
    """
    Converts a (H, W, 3) BGR colour grid into per-cell colour codes.

    Usage
    -----
    col = Colorizer(mode="truecolor")
    codes = col.fg_codes(color_bgr)   # → (H, W) object array of ANSI strings
    """

    RESET = "\x1b[0m"

    def __init__(self, mode: str = "truecolor"):
        if mode not in ("none", "ansi16", "ansi256", "truecolor", "html"):
            raise ValueError(f"Unknown color mode: {mode!r}")
        self.mode = mode

    # ------------------------------------------------------------------

    def fg_codes(self, color_bgr: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        color_bgr : (H, W, 3) uint8

        Returns
        -------
        (H, W) object array — each cell is an ANSI prefix string (or "" if none)
        """
        if self.mode == "none":
            return np.full(color_bgr.shape[:2], "", dtype=object)
        rgb = color_bgr[:, :, ::-1]   # BGR → RGB
        if self.mode == "truecolor":
            return self._truecolor_fg(rgb)
        elif self.mode == "ansi256":
            return self._ansi256_fg(rgb)
        elif self.mode == "ansi16":
            return self._ansi16_fg(rgb)
        elif self.mode == "html":
            return self._html_fg(rgb)

    def bg_codes(self, color_bgr: np.ndarray) -> np.ndarray:
        """Same as fg_codes but for background colour."""
        if self.mode == "none":
            return np.full(color_bgr.shape[:2], "", dtype=object)
        rgb = color_bgr[:, :, ::-1]
        if self.mode == "truecolor":
            return self._truecolor_bg(rgb)
        elif self.mode == "ansi256":
            return self._ansi256_bg(rgb)
        elif self.mode == "ansi16":
            return self._ansi16_bg(rgb)
        return np.full(rgb.shape[:2], "", dtype=object)

    # ------------------------------------------------------------------
    # Truecolor
    # ------------------------------------------------------------------

    @staticmethod
    def _truecolor_fg(rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        flat = rgb.reshape(-1, 3)
        codes = np.array(
            [f"\x1b[38;2;{r};{g};{b}m" for r, g, b in flat.astype(int)],
            dtype=object,
        )
        return codes.reshape(H, W)

    @staticmethod
    def _truecolor_bg(rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        flat = rgb.reshape(-1, 3)
        codes = np.array(
            [f"\x1b[48;2;{r};{g};{b}m" for r, g, b in flat.astype(int)],
            dtype=object,
        )
        return codes.reshape(H, W)

    # ------------------------------------------------------------------
    # ANSI 256
    # ------------------------------------------------------------------

    def _ansi256_fg(self, rgb: np.ndarray) -> np.ndarray:
        indices = self._nearest256(rgb)
        H, W = rgb.shape[:2]
        codes = np.array(
            [f"\x1b[38;5;{i}m" for i in indices.ravel()], dtype=object
        )
        return codes.reshape(H, W)

    def _ansi256_bg(self, rgb: np.ndarray) -> np.ndarray:
        indices = self._nearest256(rgb)
        H, W = rgb.shape[:2]
        codes = np.array(
            [f"\x1b[48;5;{i}m" for i in indices.ravel()], dtype=object
        )
        return codes.reshape(H, W)

    @staticmethod
    def _nearest256(rgb: np.ndarray) -> np.ndarray:
        """Vectorised nearest xterm-256 colour index."""
        flat = rgb.reshape(-1, 3).astype(np.float32)
        diff = flat[:, None, :] - _XTERM256[None, :, :]  # (N, 256, 3)
        dist = (diff ** 2).sum(axis=2)                    # (N, 256)
        return dist.argmin(axis=1).reshape(rgb.shape[:2])

    # ------------------------------------------------------------------
    # ANSI 16
    # ------------------------------------------------------------------

    def _ansi16_fg(self, rgb: np.ndarray) -> np.ndarray:
        indices = self._nearest16(rgb)
        H, W = rgb.shape[:2]
        codes = np.array(
            [self._ansi16_fg_code(i) for i in indices.ravel()], dtype=object
        )
        return codes.reshape(H, W)

    def _ansi16_bg(self, rgb: np.ndarray) -> np.ndarray:
        indices = self._nearest16(rgb)
        H, W = rgb.shape[:2]
        codes = np.array(
            [self._ansi16_bg_code(i) for i in indices.ravel()], dtype=object
        )
        return codes.reshape(H, W)

    @staticmethod
    def _nearest16(rgb: np.ndarray) -> np.ndarray:
        flat = rgb.reshape(-1, 3).astype(np.float32)
        diff = flat[:, None, :] - _ANSI16_PALETTE[None, :, :]
        dist = (diff ** 2).sum(axis=2)
        return dist.argmin(axis=1).reshape(rgb.shape[:2])

    @staticmethod
    def _ansi16_fg_code(idx: int) -> str:
        if idx < 8:
            return f"\x1b[{30 + idx}m"
        return f"\x1b[{90 + idx - 8}m"

    @staticmethod
    def _ansi16_bg_code(idx: int) -> str:
        if idx < 8:
            return f"\x1b[{40 + idx}m"
        return f"\x1b[{100 + idx - 8}m"

    # ------------------------------------------------------------------
    # HTML (returns hex colour strings for CSS)
    # ------------------------------------------------------------------

    @staticmethod
    def _html_fg(rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        flat = rgb.reshape(-1, 3).astype(int)
        codes = np.array(
            [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in flat], dtype=object
        )
        return codes.reshape(H, W)

    # ------------------------------------------------------------------
    # Utility: quantise to LAB for perceptual quality
    # ------------------------------------------------------------------

    @staticmethod
    def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        """Convert (H, W, 3) uint8 RGB → (H, W, 3) float32 LAB."""
        import cv2
        bgr = rgb[:, :, ::-1].astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

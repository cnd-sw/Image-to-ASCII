"""
font_metrics.py
---------------
Determine the real width:height ratio of a monospace font cell so that
the output character grid produces the correct visual aspect ratio.

Priority order for obtaining metrics
-------------------------------------
1. User supplies a TTF/OTF font path  → render a test glyph, measure bbox.
2. Platform default monospace font    → same approach.
3. Hardcoded fallback                 → 0.5 (most terminals use ~1:2 cells).
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# Hardcoded fallback (width / height)
DEFAULT_FONT_ASPECT = 0.5

# Common system monospace font search paths (macOS / Linux / Windows)
_SYSTEM_FONTS = [
    # macOS
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    # Windows (if running via WSL or cross-platform)
    "C:/Windows/Fonts/consola.ttf",
]


@lru_cache(maxsize=32)
def get_font_aspect(font_path: Optional[str] = None, size: int = 16) -> float:
    """
    Return the width/height ratio for the given font.
    Uses freetype-py when available; falls back to hardcoded constant.

    Parameters
    ----------
    font_path : optional path to .ttf/.otf font file
    size      : point size used for measurement

    Returns
    -------
    float  width / height ratio of a single character cell
    """
    try:
        import freetype  # type: ignore
    except ImportError:
        log.debug("freetype-py not installed; using default font aspect %.2f", DEFAULT_FONT_ASPECT)
        return DEFAULT_FONT_ASPECT

    path = font_path or _find_system_font()
    if path is None:
        return DEFAULT_FONT_ASPECT

    try:
        face = freetype.Face(path)
        face.set_pixel_sizes(0, size)
        # Measure 'M' — traditionally used for em-square reference
        face.load_char("M", freetype.FT_LOAD_RENDER)
        metrics = face.glyph.metrics
        w = metrics.width >> 6      # 26.6 fixed-point → integer pixels
        h = metrics.height >> 6
        if h == 0:
            return DEFAULT_FONT_ASPECT
        aspect = w / h
        log.debug("Font %s → aspect %.3f (w=%d h=%d)", path, aspect, w, h)
        return aspect
    except Exception as exc:
        log.warning("Could not measure font aspect from %s: %s", path, exc)
        return DEFAULT_FONT_ASPECT


def _find_system_font() -> Optional[str]:
    for p in _SYSTEM_FONTS:
        if os.path.exists(p):
            return p
    return None

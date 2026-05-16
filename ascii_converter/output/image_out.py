"""
image_out.py
------------
Render an AsciiFrame to a PNG image using Pillow.

The output image uses an actual monospace font (or Pillow default) so every
character is pixel-perfect and the result can be used as a high-quality still
or a video frame.
"""

from __future__ import annotations

import os
import logging
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from ascii_converter.core.renderer import AsciiFrame

log = logging.getLogger(__name__)


class ImageOutput:
    """
    Renders an AsciiFrame to a PNG file.

    Parameters
    ----------
    font_path      : path to a monospace TTF/OTF font
    font_size      : point size for rendering
    bg_color       : (R,G,B) background colour, default black
    fg_color       : (R,G,B) foreground fallback when no colour codes
    padding        : pixels of padding around the image
    use_cell_color : use the per-cell BGR colour from the frame
    """

    def __init__(
        self,
        font_path: Optional[str] = None,
        font_size: int = 14,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        fg_color: Tuple[int, int, int] = (200, 200, 200),
        padding: int = 8,
        use_cell_color: bool = True,
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.padding = padding
        self.use_cell_color = use_cell_color
        self._font = None
        self._cell_w = 0
        self._cell_h = 0
        self._init_font()

    # ------------------------------------------------------------------

    def save(self, frame: AsciiFrame, path: str) -> None:
        """Render frame and save to `path` (PNG)."""
        from PIL import Image, ImageDraw

        H, W = frame.rows, frame.cols
        img_w = W * self._cell_w + 2 * self.padding
        img_h = H * self._cell_h + 2 * self.padding

        canvas = Image.new("RGB", (img_w, img_h), self.bg_color)
        draw = ImageDraw.Draw(canvas)

        for r in range(H):
            for c in range(W):
                ch = frame.chars[r, c]
                x = self.padding + c * self._cell_w
                y = self.padding + r * self._cell_h

                if self.use_cell_color:
                    b, g, rv = frame.color[r, c]
                    color = (int(rv), int(g), int(b))
                else:
                    color = self.fg_color

                draw.text((x, y), ch, fill=color, font=self._font)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        canvas.save(path)
        log.info("Saved image → %s  (%dx%d px)", path, img_w, img_h)

    def render_to_array(self, frame: AsciiFrame) -> np.ndarray:
        """Return the rendered image as (H, W, 3) uint8 RGB numpy array."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        self.save(frame, tmp)
        from PIL import Image
        arr = np.array(Image.open(tmp).convert("RGB"))
        os.unlink(tmp)
        return arr

    # ------------------------------------------------------------------

    def _init_font(self) -> None:
        from PIL import ImageFont, Image, ImageDraw

        fallbacks = [
            self.font_path,
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        font = None
        for p in fallbacks:
            if p and os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, self.font_size)
                    log.debug("ImageOutput using font: %s", p)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()
            log.warning("ImageOutput: no TTF font found, using Pillow default.")

        self._font = font

        # Measure cell dimensions using 'M'
        dummy = Image.new("L", (100, 100))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), "M", font=font)
        self._cell_w = max(1, bbox[2] - bbox[0])
        self._cell_h = max(1, bbox[3] - bbox[1])
        log.debug("Cell size: %dx%d px", self._cell_w, self._cell_h)

"""
gif_out.py
----------
Export a list of AsciiFrame objects as an animated GIF using Pillow.
Each frame is rendered by ImageOutput, then assembled into a GIF.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
from ascii_converter.core.renderer import AsciiFrame
from ascii_converter.output.image_out import ImageOutput

log = logging.getLogger(__name__)


class GifOutput:
    """
    Saves animated GIF from a sequence of AsciiFrames.

    Parameters
    ----------
    fps          : playback frame rate (default 12)
    loop         : 0 = loop forever, 1 = play once
    optimize     : enable Pillow GIF palette optimization
    font_path    : path to monospace TTF font
    font_size    : point size
    bg_color     : background RGB tuple
    """

    def __init__(
        self,
        fps: float = 12.0,
        loop: int = 0,
        optimize: bool = True,
        font_path: Optional[str] = None,
        font_size: int = 10,
        bg_color: tuple = (0, 0, 0),
    ):
        self.fps = fps
        self.loop = loop
        self.optimize = optimize
        self._img_out = ImageOutput(
            font_path=font_path,
            font_size=font_size,
            bg_color=bg_color,
            use_cell_color=True,
        )

    # ------------------------------------------------------------------

    def save(self, frames: List[AsciiFrame], path: str) -> None:
        from PIL import Image

        if not frames:
            raise ValueError("No frames to export.")

        log.info("Rendering %d frames for GIF...", len(frames))
        pil_frames = []
        for i, f in enumerate(frames):
            arr = self._img_out.render_to_array(f)   # (H, W, 3) uint8 RGB
            pil_frames.append(Image.fromarray(arr).convert("P", palette=Image.ADAPTIVE))
            if (i + 1) % 10 == 0:
                log.debug("  rendered frame %d / %d", i + 1, len(frames))

        duration_ms = int(1000 / self.fps)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        pil_frames[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=self.loop,
            optimize=self.optimize,
        )
        log.info("Saved GIF → %s  (%d frames, %d ms/frame)", path, len(frames), duration_ms)

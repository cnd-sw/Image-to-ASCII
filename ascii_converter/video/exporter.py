"""
video/exporter.py
-----------------
Export a sequence of AsciiFrames to:
  - MP4 video (via FFmpeg, with optional original audio)
  - Animated GIF (via Pillow, delegated to gif_out.py)

Requires ffmpeg to be installed on the system PATH.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np

from ascii_converter.core.renderer import AsciiFrame
from ascii_converter.output.image_out import ImageOutput
from ascii_converter.output.gif_out import GifOutput

log = logging.getLogger(__name__)


class VideoExporter:
    """
    Exports ASCII video to MP4 or GIF.

    Parameters
    ----------
    fps       : output FPS
    font_path : monospace TTF font for rendering frames to pixel images
    font_size : point size
    bg_color  : background RGB tuple
    crf       : FFmpeg CRF (quality, lower=better; 18-28 typical)
    preset    : FFmpeg speed preset
    """

    def __init__(
        self,
        fps: float = 12.0,
        font_path: Optional[str] = None,
        font_size: int = 10,
        bg_color: tuple = (0, 0, 0),
        crf: int = 23,
        preset: str = "medium",
    ):
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self._img_out = ImageOutput(
            font_path=font_path,
            font_size=font_size,
            bg_color=bg_color,
            use_cell_color=True,
        )
        self._gif_out = GifOutput(
            fps=fps,
            font_path=font_path,
            font_size=font_size,
            bg_color=bg_color,
        )

    # ------------------------------------------------------------------

    def export_mp4(
        self,
        frames: List[AsciiFrame],
        output_path: str,
        audio_source: Optional[str] = None,
    ) -> None:
        """
        Render frames to PNG images then encode to MP4 with FFmpeg.
        Optionally copies audio track from `audio_source` (original video).
        """
        if not frames:
            raise ValueError("No frames to export.")

        self._check_ffmpeg()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="ascii_mp4_") as tmpdir:
            log.info("Rendering %d frames to temp images...", len(frames))
            pad = len(str(len(frames)))
            for i, f in enumerate(frames):
                png_path = os.path.join(tmpdir, f"frame_{i:0{pad}d}.png")
                self._img_out.save(f, png_path)
                if (i + 1) % 50 == 0:
                    log.debug("  %d / %d frames rendered", i + 1, len(frames))

            pattern = os.path.join(tmpdir, f"frame_%0{pad}d.png")
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", pattern,
            ]

            if audio_source:
                cmd += ["-i", audio_source, "-c:a", "aac", "-shortest"]

            cmd += [
                "-c:v", "libx264",
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-pix_fmt", "yuv420p",
                output_path,
            ]

            log.info("Running FFmpeg: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed (code {result.returncode}):\n{result.stderr}"
                )

        log.info("MP4 saved → %s", output_path)

    def export_gif(self, frames: List[AsciiFrame], output_path: str) -> None:
        """Export as animated GIF."""
        self._gif_out.save(frames, output_path)

    # ------------------------------------------------------------------

    @staticmethod
    def _check_ffmpeg() -> None:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            raise EnvironmentError(
                "ffmpeg not found on PATH. Install it: https://ffmpeg.org"
            )

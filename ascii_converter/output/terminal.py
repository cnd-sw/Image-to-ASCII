"""
terminal.py
-----------
Live terminal output with real-time preview support.

Features
--------
  - ANSI truecolor / 256 / 16 rendering
  - Clears and redraws in-place for video / webcam mode
  - Respects terminal size and auto-scales if requested
"""

from __future__ import annotations

import os
import sys
import shutil
import time
from typing import Optional

import numpy as np
from ascii_converter.core.renderer import AsciiFrame


class TerminalOutput:
    """
    Renders AsciiFrame objects to the terminal.

    Parameters
    ----------
    use_bg        : also colour cell backgrounds
    clear_between : clear screen between frames (video mode)
    fps_display   : show FPS counter in top-right corner
    """

    def __init__(
        self,
        use_bg: bool = False,
        clear_between: bool = False,
        fps_display: bool = False,
    ):
        self.use_bg = use_bg
        self.clear_between = clear_between
        self.fps_display = fps_display
        self._last_time: float = 0.0
        self._frame_count: int = 0

    # ------------------------------------------------------------------

    def display(self, frame: AsciiFrame) -> None:
        """Write frame to stdout."""
        if self.clear_between:
            self._move_home()

        text = frame.to_ansi_string(use_bg=self.use_bg)

        if self.fps_display:
            now = time.time()
            fps = 1.0 / (now - self._last_time) if self._last_time else 0.0
            self._last_time = now
            self._frame_count += 1
            fps_str = f" FPS: {fps:5.1f} | Frame: {self._frame_count} "
            text = self._inject_fps(text, fps_str)

        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def print_plain(self, frame: AsciiFrame) -> None:
        """Write plain text (no colour codes) to stdout."""
        print(frame.to_plain_string())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move_home() -> None:
        """Move cursor to top-left without clearing (flicker-free)."""
        sys.stdout.write("\x1b[H")

    @staticmethod
    def terminal_size() -> tuple:
        """Return (cols, rows) of the current terminal."""
        return shutil.get_terminal_size(fallback=(80, 24))

    @staticmethod
    def _inject_fps(text: str, fps_str: str) -> str:
        """Overlay FPS string at the start of the first line."""
        lines = text.split("\n")
        if lines:
            lines[0] = fps_str + lines[0][len(fps_str):]
        return "\n".join(lines)

    @staticmethod
    def clear_screen() -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    @staticmethod
    def hide_cursor() -> None:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()

    @staticmethod
    def show_cursor() -> None:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

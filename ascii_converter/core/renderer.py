"""
renderer.py
-----------
Main rendering pipeline.  Ties together:
  Preprocessor → Dithering → CharMatcher → Colorizer → AsciiFrame

An AsciiFrame is a simple dataclass holding:
  - chars    (H, W) object array of character strings
  - fg_codes (H, W) object array of ANSI/HTML colour prefixes
  - bg_codes (H, W) object array of ANSI/HTML bg codes
  - luma     (H, W) float32
  - color    (H, W, 3) uint8 BGR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ascii_converter.core.preprocessor import Preprocessor, PreprocessConfig
from ascii_converter.core.char_library import CharLibrary, CharLibraryBuilder
from ascii_converter.core.char_matcher import CharMatcher, MatchMode
from ascii_converter.core.colorizer import Colorizer
from ascii_converter.core.dithering import apply_dithering, DitherMethod

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame data container
# ---------------------------------------------------------------------------

@dataclass
class AsciiFrame:
    chars: np.ndarray       # (H, W) object — character strings
    fg_codes: np.ndarray    # (H, W) object — ANSI fg prefix or hex
    bg_codes: np.ndarray    # (H, W) object — ANSI bg prefix
    luma: np.ndarray        # (H, W) float32
    color: np.ndarray       # (H, W, 3) uint8 BGR
    rows: int
    cols: int

    def to_ansi_string(self, use_bg: bool = False) -> str:
        """Render as a single ANSI-coloured string."""
        reset = "\x1b[0m"
        lines = []
        for r in range(self.rows):
            row_parts = []
            for c in range(self.cols):
                fg = self.fg_codes[r, c]
                bg = self.bg_codes[r, c] if use_bg else ""
                ch = self.chars[r, c]
                row_parts.append(f"{bg}{fg}{ch}{reset}")
            lines.append("".join(row_parts))
        return "\n".join(lines)

    def to_plain_string(self) -> str:
        """Render as plain text (no colour codes)."""
        lines = []
        for r in range(self.rows):
            lines.append("".join(self.chars[r, :]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Renderer configuration
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    # ---- Preprocessing ----
    cols: int = 120
    rows: Optional[int] = None
    font_aspect: float = 0.5
    luma_formula: str = "perceived"
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    hue_shift: float = 0.0
    edge_boost: float = 0.3
    edge_method: str = "canny"
    canny_low: int = 50
    canny_high: int = 150
    bilateral: bool = False
    invert: bool = False
    normalize: bool = True

    # ---- Character library ----
    chars: str = "standard"
    font_path: Optional[str] = None
    font_size: int = 14
    patch_h: int = 16
    patch_w: int = 8

    # ---- Matching ----
    match_mode: MatchMode = "hybrid"
    topk: int = 8
    edge_weight: float = 0.4

    # ---- Dithering ----
    dither: DitherMethod = "floyd_steinberg"

    # ---- Colour ----
    color_mode: str = "truecolor"
    use_bg_color: bool = False

    # ---- Misc ----
    force_rebuild_library: bool = False


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    """
    High-level image → AsciiFrame converter.

    Usage
    -----
    r = Renderer(RenderConfig(cols=160, color_mode="truecolor"))
    frame = r.render(img_bgr)
    print(frame.to_ansi_string())
    """

    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg
        self._pre = self._build_preprocessor()
        self._lib = self._build_library()
        self._matcher = CharMatcher(
            self._lib,
            mode=cfg.match_mode,
            topk=cfg.topk,
            edge_weight=cfg.edge_weight,
        )
        self._colorizer = Colorizer(mode=cfg.color_mode)
        log.info(
            "Renderer ready: %dx? cols, mode=%s, color=%s",
            cfg.cols, cfg.match_mode, cfg.color_mode,
        )

    # ------------------------------------------------------------------

    def render(self, img_bgr: np.ndarray) -> AsciiFrame:
        """Convert a BGR uint8 image to an AsciiFrame."""
        cfg = self.cfg

        # 1. Preprocess
        luma, color, edges = self._pre.process(img_bgr)
        H, W = luma.shape

        # 2. Dither
        if cfg.dither != "none":
            luma = apply_dithering(luma, method=cfg.dither, levels=len(self._lib))

        # 3. Match characters
        chars = self._matcher.match(luma, edges)

        # 4. Colour codes
        fg = self._colorizer.fg_codes(color)
        bg = self._colorizer.bg_codes(color)

        return AsciiFrame(
            chars=chars,
            fg_codes=fg,
            bg_codes=bg,
            luma=luma,
            color=color,
            rows=H,
            cols=W,
        )

    # ------------------------------------------------------------------

    def update_config(self, **kwargs) -> None:
        """Partially update the config and rebuild affected components."""
        rebuild_lib = any(k in kwargs for k in (
            "chars", "font_path", "font_size", "patch_h", "patch_w",
        ))
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        self._pre = self._build_preprocessor()
        if rebuild_lib:
            self._lib = self._build_library()
        self._matcher = CharMatcher(
            self._lib,
            mode=self.cfg.match_mode,
            topk=self.cfg.topk,
            edge_weight=self.cfg.edge_weight,
        )
        self._colorizer = Colorizer(mode=self.cfg.color_mode)

    # ------------------------------------------------------------------

    def _build_preprocessor(self) -> Preprocessor:
        cfg = self.cfg
        pre_cfg = PreprocessConfig(
            cols=cfg.cols,
            rows=cfg.rows,
            font_aspect=cfg.font_aspect,
            luma_formula=cfg.luma_formula,
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            gamma=cfg.gamma,
            saturation=cfg.saturation,
            hue_shift=cfg.hue_shift,
            edge_boost=cfg.edge_boost,
            edge_method=cfg.edge_method,
            canny_low=cfg.canny_low,
            canny_high=cfg.canny_high,
            bilateral=cfg.bilateral,
            invert=cfg.invert,
            normalize=cfg.normalize,
        )
        return Preprocessor(pre_cfg)

    def _build_library(self) -> CharLibrary:
        builder = CharLibraryBuilder(
            charset_name=self.cfg.chars,
            patch_h=self.cfg.patch_h,
            patch_w=self.cfg.patch_w,
            font_path=self.cfg.font_path,
            font_size=self.cfg.font_size,
        )
        return builder.build(force_rebuild=self.cfg.force_rebuild_library)

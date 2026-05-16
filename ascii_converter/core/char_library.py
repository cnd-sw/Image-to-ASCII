"""
char_library.py
---------------
Builds and caches a library of rendered character glyphs used for matching.

Each character is rasterised into a small grayscale patch (default 8×16 px)
using Pillow.  Metadata per character:
  - density   : mean pixel intensity (0=empty, 1=full)
  - patch     : (H, W) float32 normalised bitmap
  - hog_feat  : HoG feature vector for structure matching
  - char       : the Unicode character string

Character sets
--------------
  standard      — printable ASCII 32-126
  extended      — ASCII + Latin-1 supplement
  unicode_block — ASCII + box-drawing + block elements + geometric shapes
  braille       — 256 braille patterns U+2800-U+28FF
  block         — half/quarter block Unicode glyphs
  minimal       — space + 10 density levels
  digits        — 0-9 + a few punctuation marks
"""

from __future__ import annotations

import os
import pickle
import hashlib
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character set definitions
# ---------------------------------------------------------------------------

_CHARSET_STANDARD = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
)

_CHARSET_EXTENDED = _CHARSET_STANDARD + "".join(
    chr(c) for c in range(160, 256)
)

_CHARSET_BOX = "".join(chr(c) for c in range(0x2500, 0x2580))  # box drawing
_CHARSET_BLOCK = "".join(chr(c) for c in range(0x2580, 0x25A0))  # block elements
_CHARSET_GEOM = "".join(chr(c) for c in range(0x25A0, 0x25C0))

_CHARSET_UNICODE_BLOCK = _CHARSET_STANDARD + _CHARSET_BOX + _CHARSET_BLOCK + _CHARSET_GEOM

_CHARSET_BRAILLE = "".join(chr(c) for c in range(0x2800, 0x2900))

_CHARSET_BLOCK_ONLY = " ▀▄█▌▐░▒▓" + _CHARSET_BLOCK

_CHARSET_MINIMAL = " .:-=+*#%@"

_CHARSET_DIGITS = "0123456789!@#$%^&*(),./:;"

CHARSETS = {
    "standard": _CHARSET_STANDARD,
    "extended": _CHARSET_EXTENDED,
    "unicode_block": _CHARSET_UNICODE_BLOCK,
    "braille": _CHARSET_BRAILLE,
    "block": _CHARSET_BLOCK_ONLY,
    "minimal": _CHARSET_MINIMAL,
    "digits": _CHARSET_DIGITS,
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CharEntry:
    char: str
    density: float          # mean luminance [0, 1]
    patch: np.ndarray       # (H, W) float32
    hog_feat: np.ndarray    # 1-D float32 feature vector


@dataclass
class CharLibrary:
    entries: List[CharEntry]
    patch_h: int
    patch_w: int
    # Pre-built arrays for fast vectorised lookup
    densities: np.ndarray = field(init=False)     # (N,) float32
    patches: np.ndarray = field(init=False)        # (N, H*W) float32
    hog_feats: np.ndarray = field(init=False)      # (N, F) float32
    chars: List[str] = field(init=False)

    def __post_init__(self):
        self.chars = [e.char for e in self.entries]
        self.densities = np.array([e.density for e in self.entries], dtype=np.float32)
        self.patches = np.stack(
            [e.patch.ravel() for e in self.entries], axis=0
        ).astype(np.float32)
        self.hog_feats = np.stack(
            [e.hog_feat for e in self.entries], axis=0
        ).astype(np.float32)

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CharLibraryBuilder:
    """
    Renders characters into small grayscale patches using Pillow and computes
    density + HoG features.  Results are cached to disk as a pickle file so
    subsequent runs are near-instant.
    """

    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ascii_converter")

    def __init__(
        self,
        charset_name: str = "standard",
        patch_h: int = 16,
        patch_w: int = 8,
        font_path: Optional[str] = None,
        font_size: int = 14,
    ):
        self.charset_name = charset_name
        self.charset = CHARSETS.get(charset_name, charset_name)  # allow raw string
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.font_path = font_path
        self.font_size = font_size

    # ------------------------------------------------------------------

    def build(self, force_rebuild: bool = False) -> CharLibrary:
        cache_path = self._cache_path()
        if not force_rebuild and os.path.exists(cache_path):
            log.info("Loading character library from cache: %s", cache_path)
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        log.info(
            "Building character library: charset=%s  patch=%dx%d  font=%s",
            self.charset_name,
            self.patch_w,
            self.patch_h,
            self.font_path or "system default",
        )
        entries = self._render_all()
        lib = CharLibrary(entries=entries, patch_h=self.patch_h, patch_w=self.patch_w)

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(lib, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Cached character library → %s  (%d chars)", cache_path, len(lib))
        return lib

    # ------------------------------------------------------------------

    def _render_all(self) -> List[CharEntry]:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        font = self._load_font()
        entries: List[CharEntry] = []

        for ch in self.charset:
            patch = self._render_char(ch, font)
            density = float(patch.mean())
            hog = self._hog_features(patch)
            entries.append(CharEntry(char=ch, density=density, patch=patch, hog_feat=hog))

        # Sort by density ascending so index maps naturally to brightness levels
        entries.sort(key=lambda e: e.density)
        return entries

    def _render_char(self, ch: str, font) -> np.ndarray:
        """Render a single character → (H, W) float32 [0,1]."""
        from PIL import Image, ImageDraw

        img = Image.new("L", (self.patch_w, self.patch_h), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ch, fill=255, font=font)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _load_font(self):
        """Load a Pillow ImageFont, falling back gracefully."""
        from PIL import ImageFont

        if self.font_path and os.path.exists(self.font_path):
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except Exception as exc:
                log.warning("Could not load font %s: %s", self.font_path, exc)

        # Try common system monospace fonts
        fallbacks = [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        for p in fallbacks:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, self.font_size)
                except Exception:
                    continue

        log.warning("No TTF font found; using Pillow default bitmap font.")
        return ImageFont.load_default()

    @staticmethod
    def _hog_features(patch: np.ndarray) -> np.ndarray:
        """
        Compute a lightweight HoG-like feature vector from a grayscale patch.
        Uses skimage if available; falls back to a simple gradient histogram.
        """
        try:
            from skimage.feature import hog  # type: ignore
            feat = hog(
                patch,
                orientations=8,
                pixels_per_cell=(4, 4),
                cells_per_block=(1, 1),
                visualize=False,
                feature_vector=True,
            )
            return feat.astype(np.float32)
        except ImportError:
            pass

        # Fallback: simple gradient magnitude + angle histogram (8 bins)
        gy = np.gradient(patch, axis=0)
        gx = np.gradient(patch, axis=1)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx)  # -π … π
        bins = np.linspace(-np.pi, np.pi, 9)
        hist, _ = np.histogram(ang, bins=bins, weights=mag)
        norm = hist.max()
        return (hist / norm if norm > 0 else hist).astype(np.float32)

    def _cache_path(self) -> str:
        key = f"{self.charset_name}_{self.patch_h}x{self.patch_w}_{self.font_path}_{self.font_size}"
        digest = hashlib.md5(key.encode()).hexdigest()[:12]
        return os.path.join(self.CACHE_DIR, f"charlib_{digest}.pkl")

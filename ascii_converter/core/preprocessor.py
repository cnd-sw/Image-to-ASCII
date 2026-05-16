"""
preprocessor.py
---------------
Image preprocessing pipeline.

Handles:
  - Intelligent resizing with real font-aspect-ratio correction
  - Multiple luminance formulas
  - Contrast / brightness / gamma / saturation / hue adjustments
  - Edge enhancement (Canny + Sobel blend, controllable boost)
  - Bilateral / guided filtering for content-aware smoothing
  - All operations are fully vectorised with NumPy / OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Parameters that govern the preprocessing pipeline."""

    # --- Output dimensions ---
    cols: int = 120                  # Target character columns
    rows: Optional[int] = None       # None → auto from aspect ratio
    font_aspect: float = 0.5         # char_width / char_height (monospace ~0.5)

    # --- Luminance formula ---
    # "luma"      → 0.299R + 0.587G + 0.114B  (Rec.601)
    # "perceived" → 0.2126R + 0.7152G + 0.0722B (Rec.709)
    # "minmax"    → (max(R,G,B) + min(R,G,B)) / 2  (HSL lightness)
    # "average"   → (R+G+B)/3
    # "custom"    → uses luma_weights
    luma_formula: str = "perceived"
    luma_weights: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)

    # --- Tone adjustments ---
    brightness: float = 0.0          # additive offset in [−1, 1]
    contrast: float = 1.0            # multiplicative factor
    gamma: float = 1.0               # power-law gamma (>1 → darker)
    saturation: float = 1.0          # 1.0 = unchanged, 0 = greyscale, 2 = vivid
    hue_shift: float = 0.0           # degrees, − 180…180

    # --- Edge enhancement ---
    edge_boost: float = 0.0          # 0 = off, 1 = strong enhancement
    edge_method: str = "canny"       # "canny" | "sobel" | "laplacian"
    canny_low: int = 50
    canny_high: int = 150

    # --- Smoothing ---
    bilateral: bool = False          # Apply bilateral filter before mapping
    bilateral_d: int = 5
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # --- Misc ---
    invert: bool = False             # Invert luminance (dark bg)
    normalize: bool = True           # Stretch histogram to [0, 255]


# ---------------------------------------------------------------------------
# Main Preprocessor class
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Transforms a raw BGR image (numpy array, uint8) into the data needed by
    the character matcher:
      - `luma`  : (rows, cols) float32 in [0, 1]  — luminance map
      - `color` : (rows, cols, 3) uint8 BGR        — colour map at char resolution
      - `edges` : (rows, cols) float32 in [0, 1]  — edge strength map
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        img_bgr : (H, W, 3) uint8 BGR image

        Returns
        -------
        luma   : (rows, cols) float32  in [0, 1]
        color  : (rows, cols, 3) uint8 BGR
        edges  : (rows, cols) float32  in [0, 1]
        """
        cfg = self.cfg

        # 1. HSV colour adjustments (saturation + hue) on full-res image
        img_bgr = self._adjust_hsv(img_bgr)

        # 2. Resize to target character grid dimensions
        target_w, target_h = self._compute_target_size(img_bgr.shape)
        img_small = cv2.resize(img_bgr, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)

        # 3. Store colour map (will be returned as-is for colourising)
        color = img_small.copy()

        # 4. Optional bilateral smoothing (helps in noisy images)
        if cfg.bilateral:
            img_small = cv2.bilateralFilter(
                img_small,
                cfg.bilateral_d,
                cfg.bilateral_sigma_color,
                cfg.bilateral_sigma_space,
            )

        # 5. Compute luminance
        luma = self._luminance(img_small)

        # 6. Brightness / contrast / gamma
        luma = self._tone_map(luma)

        # 7. Optional histogram normalisation
        if cfg.normalize:
            lo, hi = luma.min(), luma.max()
            if hi > lo:
                luma = (luma - lo) / (hi - lo)

        # 8. Invert if needed
        if cfg.invert:
            luma = 1.0 - luma

        # 9. Edge map
        edges = self._edge_map(img_small)

        # 10. Edge-enhanced luma
        if cfg.edge_boost > 0.0:
            luma = np.clip(luma + cfg.edge_boost * edges, 0.0, 1.0)

        return luma, color, edges

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_target_size(self, shape: tuple) -> Tuple[int, int]:
        """
        Compute (pixel_width, pixel_height) of the downsampled image so that
        the resulting character grid has cfg.cols columns and the correct number
        of rows accounting for font aspect ratio.
        """
        h, w = shape[:2]
        cfg = self.cfg
        cols = cfg.cols

        # Each character cell covers (cell_px_w) × (cell_px_h) pixels.
        # We want: cols == w / cell_px_w
        # And the displayed height (rows * cell_px_h * font_aspect) ≈
        # displayed width (cols * cell_px_w).
        # With font_aspect = cell_px_w / cell_px_h, rows ≈ (h / w) * cols / font_aspect
        if cfg.rows is not None:
            rows = cfg.rows
        else:
            rows = max(1, round((h / w) * cols * cfg.font_aspect))

        return cols, rows

    def _luminance(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 → float32 luminance in [0, 1]."""
        f = img.astype(np.float32) / 255.0
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        formula = self.cfg.luma_formula

        if formula == "luma":
            return 0.299 * r + 0.587 * g + 0.114 * b
        elif formula == "perceived":
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        elif formula == "minmax":
            return (np.maximum(np.maximum(r, g), b) +
                    np.minimum(np.minimum(r, g), b)) / 2.0
        elif formula == "average":
            return (r + g + b) / 3.0
        elif formula == "custom":
            wr, wg, wb = self.cfg.luma_weights
            return wr * r + wg * g + wb * b
        else:
            raise ValueError(f"Unknown luma_formula: {formula!r}")

    def _tone_map(self, luma: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, gamma corrections."""
        cfg = self.cfg
        # contrast (centred at 0.5)
        luma = (luma - 0.5) * cfg.contrast + 0.5
        # brightness offset
        luma = luma + cfg.brightness
        luma = np.clip(luma, 0.0, 1.0)
        # gamma
        if cfg.gamma != 1.0:
            luma = np.power(luma, cfg.gamma)
        return luma

    def _adjust_hsv(self, img: np.ndarray) -> np.ndarray:
        """Apply saturation and hue-shift in HSV space."""
        cfg = self.cfg
        if cfg.saturation == 1.0 and cfg.hue_shift == 0.0:
            return img  # fast path

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        # hue in [0, 180) in OpenCV
        if cfg.hue_shift != 0.0:
            hsv[:, :, 0] = (hsv[:, :, 0] + cfg.hue_shift / 2.0) % 180.0
        # saturation
        if cfg.saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _edge_map(self, img: np.ndarray) -> np.ndarray:
        """
        Compute a [0, 1] edge-strength map using the configured method.
        """
        cfg = self.cfg
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cfg.edge_method == "canny":
            edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high).astype(np.float32)
            edges /= 255.0
        elif cfg.edge_method == "sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(gx ** 2 + gy ** 2)
            mx = edges.max()
            edges = edges / mx if mx > 0 else edges
        elif cfg.edge_method == "laplacian":
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            edges = np.abs(lap)
            mx = edges.max()
            edges = edges / mx if mx > 0 else edges
        else:
            edges = np.zeros(gray.shape, dtype=np.float32)

        return edges

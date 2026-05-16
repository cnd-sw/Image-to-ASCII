"""
char_matcher.py
---------------
Character matching algorithms.

Modes
-----
  density   — fast luminance-only nearest-index look-up  (O(1) per pixel)
  structure — patch-based HoG cosine similarity           (O(N) per pixel)
  hybrid    — density pre-filter + structure re-rank on top-K candidates
  ml        — scikit-learn kNN trained on pre-rendered patches (best quality)

All public functions operate on numpy arrays and avoid Python-level loops
wherever possible.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Literal

from ascii_converter.core.char_library import CharLibrary

log = logging.getLogger(__name__)

MatchMode = Literal["density", "structure", "hybrid", "ml"]

# Number of orientation bins used for the simple gradient-proxy feature.
_GRAD_BINS = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CharMatcher:
    """
    Maps a (rows, cols) luminance map + (rows, cols) edge map to a 2-D grid
    of characters using the selected algorithm.
    """

    def __init__(
        self,
        library: CharLibrary,
        mode: MatchMode = "hybrid",
        topk: int = 8,
        edge_weight: float = 0.5,
    ):
        self.lib = library
        self.mode = mode
        self.topk = topk
        self.edge_weight = edge_weight
        self._knn = None

        # Normalise HoG feature matrix once — shape (N, F)
        norms = np.linalg.norm(library.hog_feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._hog_normed = library.hog_feats / norms  # (N, F)

        # Library HoG dimensionality (8 or 64+ depending on skimage availability)
        self._hog_F = self._hog_normed.shape[1]

        if mode == "ml":
            self._build_knn()

    # ------------------------------------------------------------------

    def match(
        self,
        luma: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Return a (H, W) object array of character strings."""
        if self.mode == "density":
            return self._match_density(luma)
        elif self.mode == "structure":
            return self._match_structure(luma, edges)
        elif self.mode == "hybrid":
            return self._match_hybrid(luma, edges)
        elif self.mode == "ml":
            return self._match_ml(luma, edges)
        else:
            raise ValueError(f"Unknown match mode: {self.mode!r}")

    # ------------------------------------------------------------------
    # Density matching — O(1) per pixel, fully vectorised
    # ------------------------------------------------------------------

    def _match_density(self, luma: np.ndarray) -> np.ndarray:
        N = len(self.lib)
        # Map each luminance value → nearest index in sorted density array
        indices = np.searchsorted(self.lib.densities, luma.ravel(), side="left")
        indices = np.clip(indices, 0, N - 1)
        chars = np.array(self.lib.chars, dtype=object)
        return chars[indices].reshape(luma.shape)

    # ------------------------------------------------------------------
    # Per-cell gradient proxy feature — always F-dimensional
    # ------------------------------------------------------------------

    def _cell_features(self, luma: np.ndarray) -> np.ndarray:
        """
        Build (H*W, F) float32 feature matrix from gradient orientation
        histograms.  F is padded/truncated to match self._hog_F so that
        dot-products with the library matrix are always valid.
        """
        H, W = luma.shape
        F = self._hog_F
        n_cells = H * W

        gy = np.gradient(luma, axis=0)
        gx = np.gradient(luma, axis=1)
        mag = np.sqrt(gx ** 2 + gy ** 2).ravel()    # (H*W,)
        ang = np.arctan2(gy, gx).ravel()             # (H*W,) in [-π, π]

        # 8-bin orientation histogram per cell (fully vectorised)
        bins = np.linspace(-np.pi, np.pi, _GRAD_BINS + 1)
        feats_8 = np.zeros((n_cells, _GRAD_BINS), dtype=np.float32)
        bin_idx = np.clip(np.digitize(ang, bins) - 1, 0, _GRAD_BINS - 1)
        np.add.at(feats_8, (np.arange(n_cells), bin_idx), mag)

        # L∞ normalise each row
        row_max = feats_8.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        feats_8 /= row_max

        # Pad or truncate to match library HoG dimensionality F
        if F == _GRAD_BINS:
            return feats_8
        elif F > _GRAD_BINS:
            feats = np.zeros((n_cells, F), dtype=np.float32)
            feats[:, :_GRAD_BINS] = feats_8
            return feats
        else:
            return feats_8[:, :F].copy()

    # ------------------------------------------------------------------
    # Structure matching — vectorised cosine similarity
    # ------------------------------------------------------------------

    def _match_structure(
        self, luma: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        H, W = luma.shape
        chars = np.array(self.lib.chars, dtype=object)

        feats = self._cell_features(luma)            # (H*W, F)
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats /= norms

        scores = feats @ self._hog_normed.T          # (H*W, N)
        best = scores.argmax(axis=1)
        return chars[best].reshape(H, W)

    # ------------------------------------------------------------------
    # Hybrid — density pre-filter + structure re-rank on top-K
    # ------------------------------------------------------------------

    def _match_hybrid(
        self, luma: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        H, W = luma.shape
        chars = np.array(self.lib.chars, dtype=object)
        densities = self.lib.densities      # (N,) sorted
        N = len(densities)
        K = min(self.topk, N)

        # Density-based candidate indices — (H*W, K)
        effective = np.clip(luma + self.edge_weight * edges, 0.0, 1.0)
        flat_eff = effective.ravel()
        center_idx = np.searchsorted(densities, flat_eff).clip(0, N - 1)
        half_k = K // 2
        lo = np.clip(center_idx - half_k, 0, N - K)
        candidate_idx = lo[:, None] + np.arange(K)[None, :]   # (H*W, K)

        # Per-cell gradient features — (H*W, F), L2-normalised
        feats = self._cell_features(luma)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats /= norms                                         # (H*W, F)

        # Gather candidate HoG vectors — (H*W, K, F)
        hog_cands = self._hog_normed[candidate_idx]

        # Batch cosine score: einsum → (H*W, K)
        scores = np.einsum("ikf,if->ik", hog_cands, feats)
        best_k = scores.argmax(axis=1)                        # (H*W,)

        chosen = candidate_idx[np.arange(H * W), best_k]     # (H*W,)
        return chars[chosen].reshape(H, W)

    # ------------------------------------------------------------------
    # ML (kNN) matching
    # ------------------------------------------------------------------

    def _build_knn(self):
        try:
            from sklearn.neighbors import KNeighborsClassifier  # type: ignore
        except ImportError:
            log.warning("scikit-learn not available; falling back to hybrid mode.")
            self.mode = "hybrid"
            return

        X = self.lib.hog_feats
        y = np.arange(len(self.lib))
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine", algorithm="brute")
        knn.fit(X, y)
        self._knn = knn
        log.info("kNN matcher built on %d characters.", len(self.lib))

    def _match_ml(
        self, luma: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        if self._knn is None:
            return self._match_hybrid(luma, edges)

        H, W = luma.shape
        chars = np.array(self.lib.chars, dtype=object)
        feats = self._cell_features(luma)   # (H*W, F)
        pred = self._knn.predict(feats)     # (H*W,)
        return chars[pred].reshape(H, W)

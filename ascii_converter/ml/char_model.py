"""
ml/char_model.py
----------------
Optional scikit-learn Random Forest / kNN character matcher.

When trained, this replaces the HoG-cosine matcher for superior accuracy.
The model is serialised to disk (joblib) so subsequent runs skip training.

Usage
-----
  model = CharModel(library)
  model.train()           # or model.load()
  chars = model.predict(patches)   # (N, H, W) float32 → (N,) char indices
"""

from __future__ import annotations

import os
import logging
import hashlib
import numpy as np
from typing import Optional

from ascii_converter.core.char_library import CharLibrary

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ascii_converter", "ml")


class CharModel:
    """
    Trains a kNN (or Random Forest) on rendered character patches.

    Parameters
    ----------
    library    : CharLibrary with pre-rendered patches
    model_type : "knn" | "rf" (random forest)
    n_neighbors: neighbours for kNN
    n_estimators: trees for RF
    """

    def __init__(
        self,
        library: CharLibrary,
        model_type: str = "knn",
        n_neighbors: int = 1,
        n_estimators: int = 50,
    ):
        self.library = library
        self.model_type = model_type
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self._model = None
        self._cache_path = self._get_cache_path()

    # ------------------------------------------------------------------

    def train(self, force: bool = False) -> None:
        """Train and cache the model."""
        if not force and os.path.exists(self._cache_path):
            self.load()
            return

        try:
            import sklearn  # type: ignore
        except ImportError:
            raise RuntimeError("scikit-learn is required for ML mode. pip install scikit-learn")

        X = self.library.patches           # (N, H*W) float32
        y = np.arange(len(self.library))   # class indices

        log.info("Training %s on %d characters...", self.model_type.upper(), len(y))

        if self.model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                metric="euclidean",
                algorithm="ball_tree",
            )
        elif self.model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                n_jobs=-1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        model.fit(X, y)
        self._model = model
        self._save()
        log.info("Model trained and cached → %s", self._cache_path)

    def load(self) -> None:
        """Load a cached model from disk."""
        try:
            import joblib  # type: ignore
        except ImportError:
            import pickle as joblib  # type: ignore (fallback)
        self._model = joblib.load(self._cache_path)
        log.info("ML model loaded from cache: %s", self._cache_path)

    def predict_indices(self, patches: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        patches : (N, H, W) or (N, F) float32

        Returns
        -------
        (N,) int array of character indices into library.chars
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call .train() first.")
        flat = patches.reshape(len(patches), -1).astype(np.float32)
        return self._model.predict(flat).astype(int)

    # ------------------------------------------------------------------

    def _save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            import joblib
            joblib.dump(self._model, self._cache_path, compress=3)
        except ImportError:
            import pickle
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_cache_path(self) -> str:
        key = f"{self.model_type}_{len(self.library)}_{self.library.patch_h}x{self.library.patch_w}"
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        return os.path.join(CACHE_DIR, f"charmodel_{digest}.pkl")

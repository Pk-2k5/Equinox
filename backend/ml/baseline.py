"""
ml/baseline.py

Patient-specific baseline module.

Responsibilities:
  1. Accumulate feature vectors from windows labeled "normal"
  2. Compute per-feature mean and std using Welford's online algorithm
     (no need to store raw windows — O(1) memory regardless of sample count)
  3. Persist baseline stats to a JSON file so they survive restarts
  4. Normalise any incoming feature vector to deviation scores:
       normalised[i] = (feature[i] - mean[i]) / (std[i] + ε)
     This makes every feature patient-relative rather than absolute.

Minimum normal windows required before baseline is considered valid: 30
(~1 minute of calm data at one window per second)

File layout (baseline.json):
  {
    "n_windows": 45,
    "means":  [28 floats],
    "m2s":    [28 floats],   <-- Welford's M2 accumulators
    "last_updated": "ISO timestamp"
  }
  stds are derived at read-time as sqrt(M2 / (n-1)) — not stored separately
  so they're always consistent with the accumulators.
"""

import json
import os
import datetime
import threading
import numpy as np

from .features import FEATURE_NAMES

N_FEATURES     = 28
MIN_WINDOWS    = 30      # baseline is "ready" only after this many normal windows
_EPSILON       = 1e-8    # prevents division by zero in normalise()
_DEFAULT_PATH  = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "baseline.json"
)


class BaselineModule:
    """
    Welford online mean/variance accumulator for patient baseline.

    Thread-safe: update() and normalise() can be called from different threads.
    """

    def __init__(self, filepath: str = _DEFAULT_PATH):
        self._filepath = os.path.abspath(filepath)
        self._lock     = threading.Lock()

        # Welford accumulators
        self._n    = 0
        self._mean = np.zeros(N_FEATURES, dtype=np.float64)
        self._m2   = np.zeros(N_FEATURES, dtype=np.float64)

        # Try to load existing baseline from disk
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, features: np.ndarray) -> None:
        """
        Add one normal-window feature vector to the baseline.

        Parameters
        ----------
        features : np.ndarray, shape (28,)
            Output of features.extract() for a window labeled "normal".
        """
        if features.shape != (N_FEATURES,):
            raise ValueError(f"Expected (28,) feature vector, got {features.shape}")

        x = features.astype(np.float64)
        with self._lock:
            self._n += 1
            delta   = x - self._mean
            self._mean += delta / self._n
            delta2  = x - self._mean
            self._m2   += delta * delta2

        self.save()   # persist after every update so progress is never lost

    def normalise(self, features: np.ndarray) -> np.ndarray:
        """
        Convert raw feature vector to patient-relative deviation scores.

        normalised[i] = (features[i] - mean[i]) / (std[i] + ε)

        Parameters
        ----------
        features : np.ndarray, shape (28,)

        Returns
        -------
        np.ndarray, shape (28,), dtype float32
            Zero-centred scores; positive = above patient normal,
            negative = below.
        """
        with self._lock:
            mean = self._mean.copy()
            std  = self._std().copy()

        return ((features.astype(np.float64) - mean) / (std + _EPSILON)).astype(np.float32)

    def is_ready(self) -> bool:
        """True once enough normal windows have been collected."""
        with self._lock:
            return self._n >= MIN_WINDOWS

    def stats(self) -> dict:
        """Return a summary dict for logging / API exposure."""
        with self._lock:
            return {
                "n_windows":   self._n,
                "is_ready":    self._n >= MIN_WINDOWS,
                "min_required": MIN_WINDOWS,
                "feature_means": dict(zip(FEATURE_NAMES, self._mean.tolist())),
                "feature_stds":  dict(zip(FEATURE_NAMES, self._std().tolist())),
            }

    def reset(self) -> None:
        """
        Wipe the baseline entirely (use when recalibrating from scratch).
        Deletes the persisted file.
        """
        with self._lock:
            self._n    = 0
            self._mean = np.zeros(N_FEATURES, dtype=np.float64)
            self._m2   = np.zeros(N_FEATURES, dtype=np.float64)

        if os.path.exists(self._filepath):
            os.remove(self._filepath)

    def save(self) -> None:
        """Write current accumulators to JSON."""
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        with self._lock:
            payload = {
                "n_windows":    self._n,
                "means":        self._mean.tolist(),
                "m2s":          self._m2.tolist(),
                "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
            }
        with open(self._filepath, "w") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _std(self) -> np.ndarray:
        """Population-safe std from Welford M2.  Caller must hold _lock."""
        if self._n < 2:
            return np.ones(N_FEATURES, dtype=np.float64)   # fallback: no division by zero
        return np.sqrt(self._m2 / (self._n - 1))

    def _load(self) -> None:
        """Load accumulators from JSON if the file exists."""
        if not os.path.exists(self._filepath):
            return
        try:
            with open(self._filepath) as f:
                payload = json.load(f)
            self._n    = int(payload["n_windows"])
            self._mean = np.array(payload["means"],  dtype=np.float64)
            self._m2   = np.array(payload["m2s"],    dtype=np.float64)
            print(f"[baseline] Loaded baseline: {self._n} normal windows "
                  f"from {self._filepath}")
        except Exception as e:
            print(f"[baseline] Could not load baseline file: {e}. Starting fresh.")
            self._n    = 0
            self._mean = np.zeros(N_FEATURES, dtype=np.float64)
            self._m2   = np.zeros(N_FEATURES, dtype=np.float64)

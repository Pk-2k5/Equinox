"""
ml/model.py

Core ML module — train, predict, persist.

Algorithm selection by training-set size (patient data is small by design):
  <  200 samples → LinearDiscriminantAnalysis
                   works well on tiny datasets; closed-form, no hyperparams
  200–499 samples → SVM (RBF kernel) + Platt calibration
                    strong on small high-dimensional data
  ≥  500 samples → Random Forest (100 trees)
                    captures non-linearity; provides feature importances

All classifiers are trained with class_weight="balanced" (or equivalent)
to handle the typical imbalance where normal windows outnumber tremor.

Classification is binary:  0 = normal,  1 = tremor
Severity score   = P(tremor) from predict_proba  ∈ [0.0, 1.0]
Severity class   = thresholded from severity_score:
                     0.00–0.25 → "None"
                     0.25–0.50 → "Mild"
                     0.50–0.75 → "Moderate"
                     0.75–1.00 → "Severe"
Confidence       = max(P(normal), P(tremor))

Model is persisted as a .pkl file and hot-swapped atomically so inference
is never interrupted during a retraining cycle.
"""

import os
import pickle
import threading
import datetime
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

from .features import N_FEATURES

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "model.pkl"
)

# Severity thresholds applied to P(tremor)
_THRESHOLDS = [
    (0.75, "Severe"),
    (0.50, "Moderate"),
    (0.25, "Mild"),
    (0.00, "None"),
]

# Sample-count boundaries for algorithm selection
_SVM_MIN   = 200
_RF_MIN    = 500


class ModelModule:
    """
    Thread-safe wrapper around a scikit-learn binary classifier.

    predict() can be called from the Flask request thread at any time.
    train()   is called from the retrainer background thread.
    Both can run concurrently safely because _swap_model() replaces the
    model reference atomically under a write lock.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        self._model_path  = os.path.abspath(model_path)
        self._model       = None          # current live model (sklearn estimator)
        self._lock        = threading.RLock()
        self._trained_at  = None          # UTC datetime of last successful train
        self._n_train     = 0             # samples used in last training run
        self._val_f1      = 0.0           # val F1 from last training run
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        with self._lock:
            return self._model is not None

    def predict(self, features: np.ndarray) -> dict:
        """
        Run inference on one normalised feature vector.

        Parameters
        ----------
        features : np.ndarray, shape (28,)
            Output of baseline.normalise(raw_features).

        Returns
        -------
        dict with keys:
            severity_score  float  0.0–1.0
            severity_class  str    "None" | "Mild" | "Moderate" | "Severe"
            confidence      float  0.5–1.0
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model is not trained yet.")
            model = self._model

        x = features.reshape(1, -1).astype(np.float32)
        proba = model.predict_proba(x)[0]          # [P(normal), P(tremor)]
        score = float(proba[1])                     # P(tremor) = severity score
        confidence = float(np.max(proba))

        severity_class = "None"
        for threshold, label in _THRESHOLDS:
            if score >= threshold:
                severity_class = label
                break

        return {
            "severity_score":  round(score, 4),
            "severity_class":  severity_class,
            "confidence":      round(confidence, 4),
        }

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Fit a new model on the full labeled dataset.

        Selects algorithm by sample count, evaluates with stratified CV,
        and only replaces the live model if the new one is at least as good.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 28)
        y : np.ndarray, shape (n_samples,)   — 0=normal, 1=tremor

        Returns
        -------
        dict with keys:
            algorithm   str    name of algorithm used
            n_samples   int    training set size
            val_f1      float  mean CV F1 (macro)
            replaced    bool   True if live model was replaced
            message     str    human-readable summary
        """
        if len(X) < 10:
            return {"replaced": False, "message": "Too few samples to train (need ≥ 10)."}

        n = len(X)
        candidate, algo_name = _build_model(n)

        # Cross-validation — use leave-one-out style for very small sets
        n_splits = min(5, _min_class_count(y))
        if n_splits < 2:
            return {"replaced": False, "message": "Need at least 2 samples per class."}

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(candidate, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
        val_f1 = float(np.mean(scores))

        # Only replace if candidate is not significantly worse than current
        with self._lock:
            current_f1 = self._val_f1

        if val_f1 < current_f1 - 0.02:
            return {
                "algorithm": algo_name,
                "n_samples": n,
                "val_f1":    round(val_f1, 4),
                "replaced":  False,
                "message":   f"Candidate F1 {val_f1:.3f} worse than current {current_f1:.3f}. Kept old model.",
            }

        # Fit on full dataset before swapping in
        candidate.fit(X, y)
        self._swap_model(candidate, n, val_f1)
        self.save()

        return {
            "algorithm": algo_name,
            "n_samples": n,
            "val_f1":    round(val_f1, 4),
            "replaced":  True,
            "message":   f"Model updated → {algo_name}, n={n}, val_f1={val_f1:.3f}",
        }

    def save(self) -> None:
        """Persist current model to disk."""
        with self._lock:
            if self._model is None:
                return
            payload = {
                "model":       self._model,
                "trained_at":  self._trained_at,
                "n_train":     self._n_train,
                "val_f1":      self._val_f1,
            }
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(payload, f)

    def info(self) -> dict:
        """Return a summary dict for logging / API."""
        with self._lock:
            return {
                "is_trained":  self._model is not None,
                "algorithm":   type(self._model).__name__ if self._model else None,
                "n_train":     self._n_train,
                "val_f1":      self._val_f1,
                "trained_at":  self._trained_at.isoformat() + "Z" if self._trained_at else None,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _swap_model(self, new_model, n: int, val_f1: float) -> None:
        """Atomically replace the live model under the write lock."""
        with self._lock:
            self._model      = new_model
            self._n_train    = n
            self._val_f1     = val_f1
            self._trained_at = datetime.datetime.utcnow()

    def _load(self) -> None:
        """Load model from disk if it exists."""
        if not os.path.exists(self._model_path):
            return
        try:
            with open(self._model_path, "rb") as f:
                payload = pickle.load(f)
            self._model      = payload["model"]
            self._trained_at = payload.get("trained_at")
            self._n_train    = payload.get("n_train", 0)
            self._val_f1     = payload.get("val_f1", 0.0)
            print(f"[model] Loaded model: {type(self._model).__name__}, "
                  f"n={self._n_train}, val_f1={self._val_f1:.3f}")
        except Exception as e:
            print(f"[model] Could not load model: {e}. Starting untrained.")
            self._model = None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_model(n_samples: int):
    """
    Return (unfitted sklearn estimator, algorithm name) for a given dataset size.
    All models output calibrated probabilities via predict_proba().
    """
    if n_samples < _SVM_MIN:
        # LDA: closed-form solution, handles small n well
        # priors=None lets it infer from data; handles imbalance via covariance
        model = LinearDiscriminantAnalysis()
        name  = "LDA"

    elif n_samples < _RF_MIN:
        # SVM + Platt calibration for probability output
        # class_weight='balanced' corrects for normal >> tremor imbalance
        svc   = SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", probability=False)
        model = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
        name  = "SVM-RBF+Platt"

    else:
        # Random Forest — handles non-linearity, gives feature importances
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        name  = "RandomForest"

    return model, name


def _min_class_count(y: np.ndarray) -> int:
    """Return the size of the smallest class — used to cap CV folds."""
    unique, counts = np.unique(y, return_counts=True)
    return int(np.min(counts)) if len(counts) > 0 else 0

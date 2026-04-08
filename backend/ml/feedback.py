"""
ml/feedback.py

Feedback manager — label request gating, phase tracking, label routing.

Responsibilities:
  1. Track system phase:  calibration → inference
  2. Decide when to ask the user for a label (fatigue-aware gating)
  3. Hold a pending label request until the user responds
  4. Route submitted labels to baseline.py and storage.py

─────────────────────────────────────────────────
Phase transitions
─────────────────────────────────────────────────
  calibration  → inference   when BOTH:
                               baseline.is_ready()  (≥ 30 normal windows)
                               model.is_trained     (first train succeeded)

  inference stays inference for the rest of the session.
  There is no separate "baseline" phase exposed externally —
  the first model training fires automatically once baseline + min
  labels are available; that is handled by retrainer.py.

─────────────────────────────────────────────────
Label request gating (prevents user fatigue)
─────────────────────────────────────────────────
  A label request is issued when ALL conditions hold:
    a) No request is already pending (one at a time)
    b) At least COOLDOWN_SECONDS have passed since the last request
    c) One of:
         • We are in calibration phase  (always want more data)
         • Tremor band power of resultant acc > TREMOR_POWER_THRESHOLD
         • Model confidence < CONFIDENCE_THRESHOLD (model is unsure)

─────────────────────────────────────────────────
Label routing on submit
─────────────────────────────────────────────────
  "normal"  → baseline.update(features)   (builds patient baseline)
              storage.save_window(...)     (training data)
  "tremor"  → storage.save_window(...)    (training data only)
"""

import time
import threading
import uuid
import numpy as np

from .baseline import BaselineModule
from .model    import ModelModule
from .storage  import LabelStore
from .features import FEATURE_NAMES

# Gating constants
COOLDOWN_SECONDS       = 30     # minimum gap between label requests
TREMOR_POWER_THRESHOLD = 0.15   # resultant acc tremor band power fraction
CONFIDENCE_THRESHOLD   = 0.55   # request label when model is below this

# Feature indices used for gating decisions
_TBP_AX = FEATURE_NAMES.index("tbp_ax")   # tremor band power starts at 12
_TBP_AY = FEATURE_NAMES.index("tbp_ay")
_TBP_AZ = FEATURE_NAMES.index("tbp_az")


class FeedbackManager:
    """
    Coordinates label requests and routes answers to the right modules.

    Instantiated once at app startup, shared across threads.
    """

    def __init__(self,
                 baseline: BaselineModule,
                 model:    ModelModule,
                 store:    LabelStore):
        self._baseline = baseline
        self._model    = model
        self._store    = store
        self._lock     = threading.Lock()

        # Pending request state
        self._pending_window_id : str | None      = None
        self._pending_features  : np.ndarray | None = None
        self._pending_since     : float           = 0.0   # time.monotonic()

        # Gating state
        self._last_request_time : float = 0.0      # time.monotonic()

    # ------------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------------

    def phase(self) -> str:
        """
        Return the current system phase.

        "calibration"  — still building baseline / first model
        "inference"    — baseline ready + model trained → real-time predictions
        """
        if self._baseline.is_ready() and self._model.is_trained:
            return "inference"
        return "calibration"

    # ------------------------------------------------------------------
    # Label request gating
    # ------------------------------------------------------------------

    def check_and_request(self,
                          raw_features: np.ndarray,
                          prediction: dict | None = None) -> dict | None:
        """
        Decide whether to issue a label request for this window.

        Called after every prediction (inference) or every window
        (calibration) from the ML pipeline.

        Parameters
        ----------
        raw_features : np.ndarray, shape (28,)
            Raw (un-normalised) feature vector for this window.
        prediction   : dict | None
            Output of model.predict(), or None if in calibration phase.

        Returns
        -------
        dict   — label request payload to embed in /data JSON response, OR
        None   — no request this cycle
        """
        with self._lock:
            now = time.monotonic()

            # Don't stack requests
            if self._pending_window_id is not None:
                return None

            # Enforce cooldown
            if (now - self._last_request_time) < COOLDOWN_SECONDS:
                return None

            # Check trigger conditions
            triggered = False

            if self.phase() == "calibration":
                # During calibration always want labeled data
                triggered = True

            else:
                # Inference phase: only ask when signal looks tremor-like
                # OR model is uncertain
                tbp_resultant = float(np.mean(raw_features[[_TBP_AX, _TBP_AY, _TBP_AZ]]))
                if tbp_resultant > TREMOR_POWER_THRESHOLD:
                    triggered = True
                elif prediction and prediction.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
                    triggered = True

            if not triggered:
                return None

            # Issue request
            window_id = _make_window_id()
            self._pending_window_id = window_id
            self._pending_features  = raw_features.copy()
            self._pending_since     = now
            self._last_request_time = now

        return {
            "requested":  True,
            "window_id":  window_id,
            "prompt":     "Movement detected. Was this involuntary tremor or normal movement?",
            "options":    ["normal", "tremor"],
        }

    def pending_request(self) -> dict | None:
        """
        Return the currently open label request (for embedding in /data),
        or None if none is pending.
        """
        with self._lock:
            if self._pending_window_id is None:
                return None
            return {
                "requested":  True,
                "window_id":  self._pending_window_id,
                "prompt":     "Movement detected. Was this involuntary tremor or normal movement?",
                "options":    ["normal", "tremor"],
            }

    # ------------------------------------------------------------------
    # Label submission
    # ------------------------------------------------------------------

    def submit_label(self, window_id: str, label: str) -> dict:
        """
        Accept a label from the user and route it to the right modules.

        Called by the POST /log_data route handler.

        Parameters
        ----------
        window_id : str   must match the pending request's window_id
        label     : str   "normal" | "tremor"

        Returns
        -------
        dict  {"status": "ok" | "error", "message": str}
        """
        if label not in ("normal", "tremor"):
            return {"status": "error", "message": f"Unknown label '{label}'."}

        with self._lock:
            if self._pending_window_id != window_id:
                return {
                    "status":  "error",
                    "message": f"window_id '{window_id}' does not match pending request.",
                }
            features  = self._pending_features.copy()
            source    = "calibration" if self.phase() == "calibration" else "feedback"
            # Clear pending state
            self._pending_window_id = None
            self._pending_features  = None

        # Route label — outside the lock so module calls don't block gating
        if label == "normal":
            self._baseline.update(features)

        phase_at_submit = source   # snapshot taken inside lock above
        saved = self._store.save_window(
            window_id=window_id,
            features=features,
            label=label,
            source=phase_at_submit,
        )

        if saved:
            counts = self._store.label_counts()
            return {
                "status":  "ok",
                "message": f"Label '{label}' saved. "
                           f"Dataset: {counts['normal']} normal, {counts['tremor']} tremor.",
            }
        else:
            return {"status": "error", "message": "Duplicate window_id — label not saved."}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return a summary dict for logging / API."""
        with self._lock:
            pending = self._pending_window_id is not None
            cooldown_remaining = max(
                0.0,
                COOLDOWN_SECONDS - (time.monotonic() - self._last_request_time)
            )
        return {
            "phase":              self.phase(),
            "pending_request":    pending,
            "cooldown_remaining": round(cooldown_remaining, 1),
            "baseline_ready":     self._baseline.is_ready(),
            "model_trained":      self._model.is_trained,
            **self._store.label_counts(),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_window_id() -> str:
    """Generate a short unique window ID, e.g. 'w_3f2a1c'."""
    return "w_" + uuid.uuid4().hex[:6]

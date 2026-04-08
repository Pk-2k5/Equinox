"""
ml/pipeline.py

MLPipeline — single orchestrator that wires all ML modules together.

Instantiated once at app startup.  Both app_ws.py and app.py import the
same instance via `from ml.pipeline import ml_pipeline`.

Public interface (called from Flask app):
  ml_pipeline.feed(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
      Called by process_sensor_line() on every incoming IMU frame.
      Non-blocking — just pushes into the ring buffer.

  ml_pipeline.latest_prediction() → dict
      Called by GET /data to merge ML fields into the JSON response.

  ml_pipeline.submit_label(window_id, label) → dict
      Called by POST /log_data when user answers a label request.

  ml_pipeline.start()
      Called once at app startup — launches worker + retrainer threads.

Internal flow (background worker thread, runs every 50 ms):
  buffer.get_window()          → (100, 6) window or None
  preprocessor.preprocess()   → cleaned (100, 6)
  features.extract()          → raw (28,) feature vector
  baseline.normalise()        → patient-relative (28,) vector
  model.predict()             → {severity_score, severity_class, confidence}
  feedback.check_and_request()→ decide if label needed
  → update _latest (thread-safe dict)
"""

import time
import threading
import numpy as np

from .buffer      import RingBuffer
from .preprocessor import preprocess
from .features    import extract
from .baseline    import BaselineModule
from .model       import ModelModule
from .storage     import LabelStore
from .feedback    import FeedbackManager
from .retrainer   import Retrainer

_WORKER_INTERVAL = 0.05   # seconds between buffer polls (50 ms)

# Default prediction returned before the first window is processed
_NULL_PREDICTION = {
    "severity_score":  None,
    "severity_class":  None,
    "confidence":      None,
    "phase":           "calibration",
    "label_request":   None,
}


class MLPipeline:
    """
    Orchestrates the full ML pipeline from raw IMU frames to predictions.
    Thread-safe singleton — safe to import and call from any Flask thread.
    """

    def __init__(self):
        # Instantiate all modules
        self._buffer   = RingBuffer()
        self._baseline = BaselineModule()
        self._model    = ModelModule()
        self._store    = LabelStore()
        self._feedback = FeedbackManager(self._baseline, self._model, self._store)
        self._retrainer = Retrainer(self._model, self._store)

        # Shared latest-prediction state
        self._latest      = dict(_NULL_PREDICTION)
        self._pred_lock   = threading.Lock()

        self._started = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch background worker and retrainer threads (call once)."""
        if self._started:
            return
        self._started = True

        worker = threading.Thread(
            target=self._worker_loop,
            name="ml-worker",
            daemon=True,
        )
        worker.start()

        self._retrainer.start()
        print("[pipeline] ML pipeline started.")

    # ------------------------------------------------------------------
    # Feed — called from websocket / serial reader thread
    # ------------------------------------------------------------------

    def feed(self,
             ax_g: float, ay_g: float, az_g: float,
             gx_dps: float, gy_dps: float, gz_dps: float) -> None:
        """
        Push one IMU frame into the ring buffer.
        Non-blocking — returns immediately.
        """
        self._buffer.add_sample(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)

    # ------------------------------------------------------------------
    # Read — called from /data route
    # ------------------------------------------------------------------

    def latest_prediction(self) -> dict:
        """
        Return the most recent ML output, ready to merge into /data JSON.

        Always returns a dict with these keys:
            severity_score  float | None
            severity_class  str   | None
            confidence      float | None
            phase           str   "calibration" | "inference"
            label_request   dict  | None
        """
        with self._pred_lock:
            return dict(self._latest)

    # ------------------------------------------------------------------
    # Label submission — called from /log_data route
    # ------------------------------------------------------------------

    def submit_label(self, window_id: str | None, label: str) -> dict:
        """
        Route a user label to feedback manager.

        Parameters
        ----------
        window_id : str | None  — present only for ML-triggered label requests
        label     : str         — "normal" | "tremor"

        Returns
        -------
        dict  {"status": "ok"|"error", "message": str}
        """
        if window_id is None:
            return {"status": "error", "message": "No window_id in request."}
        return self._feedback.submit_label(window_id, label)

    # ------------------------------------------------------------------
    # Status helpers (for /ml_status endpoint added to app)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "pipeline":  {"buffer_length": self._buffer.length,
                          "buffer_warm":   self._buffer.is_warm()},
            "baseline":  self._baseline.stats(),
            "model":     self._model.info(),
            "feedback":  self._feedback.status(),
            "retrainer": self._retrainer.status(),
        }

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Continuously polls the ring buffer and runs the ML pipeline
        whenever a new window is available.
        """
        while True:
            time.sleep(_WORKER_INTERVAL)
            try:
                self._process_next_window()
            except Exception as e:
                print(f"[pipeline] Worker error: {e}")

    def _process_next_window(self) -> None:
        """Process one window if ready — otherwise return immediately."""
        window = self._buffer.get_window()
        if window is None:
            return   # not enough new samples yet

        # Step 1 — clean the signal
        try:
            cleaned = preprocess(window)
        except Exception as e:
            print(f"[pipeline] Preprocess error: {e}")
            return

        # Step 2 — extract raw features
        raw_features = extract(cleaned)

        # Step 3 — normalise against patient baseline (if ready)
        if self._baseline.is_ready():
            norm_features = self._baseline.normalise(raw_features)
        else:
            norm_features = raw_features   # use raw until baseline is established

        # Step 4 — predict (only if model is trained)
        prediction = None
        if self._model.is_trained:
            try:
                prediction = self._model.predict(norm_features)
            except Exception as e:
                print(f"[pipeline] Predict error: {e}")

        # Step 5 — decide whether to request a label
        self._feedback.check_and_request(raw_features, prediction)

        # Step 6 — update shared latest-prediction (always, every window)
        phase = self._feedback.phase()
        label_request = self._feedback.pending_request()

        with self._pred_lock:
            self._latest = {
                "severity_score": prediction["severity_score"] if prediction else None,
                "severity_class": prediction["severity_class"] if prediction else None,
                "confidence":     prediction["confidence"]     if prediction else None,
                "phase":          phase,
                "label_request":  label_request,
            }


# Module-level singleton — import this everywhere
ml_pipeline = MLPipeline()

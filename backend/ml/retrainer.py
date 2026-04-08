"""
ml/retrainer.py

Background retraining scheduler.

Runs as a daemon thread alongside the Flask server.
Watches the label store and triggers model.train() when enough
new labeled data has accumulated, or when the weekly timer fires.

Trigger conditions (whichever comes first):
  • RETRAIN_EVERY_N  new labeled windows since last training run
  • RETRAIN_EVERY_DAYS  days have elapsed since last training run

Safety rules:
  • Never trains with fewer than MIN_TOTAL_SAMPLES total windows
  • Never trains if any class has fewer than MIN_PER_CLASS samples
  • model.train() itself does CV + F1 gate — retrainer just calls it
  • Keeps the last MAX_BACKUPS model .pkl snapshots on disk
  • All retrain events are appended to a log file (retrain_log.json)

The retrainer thread sleeps CHECK_INTERVAL_SECONDS between checks so it
uses negligible CPU while idle.
"""

import json
import os
import shutil
import time
import threading
import datetime

from .model   import ModelModule
from .storage import LabelStore

# Trigger thresholds
RETRAIN_EVERY_N    = 50      # new labeled windows since last train
RETRAIN_EVERY_DAYS = 7       # calendar days since last train
CHECK_INTERVAL     = 60      # seconds between store checks

# Data requirements
MIN_TOTAL_SAMPLES  = 10      # absolute floor before first train attempt
MIN_PER_CLASS      = 2       # both classes must have at least this many

# Backup settings
MAX_BACKUPS        = 3

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data"
)
_LOG_PATH = os.path.join(_DATA_DIR, "retrain_log.json")


class Retrainer:
    """
    Background retraining scheduler.

    Usage (in app startup):
        retrainer = Retrainer(model, store)
        retrainer.start()
    """

    def __init__(self, model: ModelModule, store: LabelStore):
        self._model  = model
        self._store  = store
        self._lock   = threading.Lock()
        self._thread: threading.Thread | None = None

        # Track timing of last training run
        self._last_train_time: datetime.datetime | None = None
        self._is_running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background daemon thread (call once at app startup)."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="retrainer",
            daemon=True,   # dies automatically when Flask process exits
        )
        self._thread.start()
        print("[retrainer] Background retraining thread started.")

    def force_retrain(self) -> dict:
        """
        Trigger an immediate retrain regardless of thresholds.
        Useful for manual testing or after a large bulk label upload.
        Returns the result dict from model.train().
        """
        return self._run_retrain(forced=True)

    def status(self) -> dict:
        """Return a summary for logging / API."""
        with self._lock:
            last = (self._last_train_time.isoformat() + "Z"
                    if self._last_train_time else None)
        counts = self._store.label_counts()
        return {
            "last_train_time":    last,
            "untrained_count":    self._store.untrained_count(),
            "retrain_every_n":    RETRAIN_EVERY_N,
            "retrain_every_days": RETRAIN_EVERY_DAYS,
            **counts,
        }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main loop — checks triggers every CHECK_INTERVAL seconds."""
        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                self._check_and_maybe_retrain()
            except Exception as e:
                # Never let a crash kill the retrainer thread
                print(f"[retrainer] Unexpected error during check: {e}")

    def _check_and_maybe_retrain(self) -> None:
        """Evaluate both trigger conditions and retrain if either fires."""
        new_count = self._store.untrained_count()
        triggered_n    = new_count >= RETRAIN_EVERY_N
        triggered_time = self._days_since_last_train() >= RETRAIN_EVERY_DAYS

        if triggered_n:
            print(f"[retrainer] N-trigger: {new_count} new labeled windows.")
        if triggered_time:
            print(f"[retrainer] Time-trigger: ≥{RETRAIN_EVERY_DAYS} days since last train.")

        if triggered_n or triggered_time:
            self._run_retrain(forced=False)

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def _run_retrain(self, forced: bool = False) -> dict:
        """
        Execute one retraining cycle.

        Steps:
          1. Check minimum data requirements
          2. Fetch untrained window IDs for later mark_trained()
          3. Call model.train() with the full labeled dataset
          4. If model was replaced: backup old pkl, mark windows trained, log
          5. Return result dict
        """
        counts = self._store.label_counts()

        # Guard: absolute minimum
        if counts["total"] < MIN_TOTAL_SAMPLES:
            msg = (f"[retrainer] Skipping retrain: only {counts['total']} samples "
                   f"(need ≥ {MIN_TOTAL_SAMPLES}).")
            print(msg)
            return {"replaced": False, "message": msg}

        # Guard: both classes must be present
        if counts["normal"] < MIN_PER_CLASS or counts["tremor"] < MIN_PER_CLASS:
            msg = (f"[retrainer] Skipping retrain: need ≥{MIN_PER_CLASS} per class "
                   f"(normal={counts['normal']}, tremor={counts['tremor']}).")
            print(msg)
            return {"replaced": False, "message": msg}

        # Snapshot untrained IDs BEFORE training so we know what to mark
        _, _, untrained_ids = self._store.get_untrained()

        # Fetch full dataset for training
        X, y = self._store.get_all_labeled()

        print(f"[retrainer] Starting retrain — n={len(X)}, "
              f"normal={counts['normal']}, tremor={counts['tremor']}, forced={forced}")

        result = self._model.train(X, y)
        print(f"[retrainer] Result: {result['message']}")

        if result.get("replaced"):
            self._rotate_backups()
            self._store.mark_trained(untrained_ids)
            with self._lock:
                self._last_train_time = datetime.datetime.utcnow()
            self._append_log(result, counts)

        return result

    # ------------------------------------------------------------------
    # Backup rotation
    # ------------------------------------------------------------------

    def _rotate_backups(self) -> None:
        """
        Shift existing backups down by one slot, then copy current model
        into backup slot 1.

        Slots:  model_backup_1.pkl (newest) … model_backup_N.pkl (oldest)
        """
        os.makedirs(_DATA_DIR, exist_ok=True)
        model_path = self._model._model_path

        if not os.path.exists(model_path):
            return

        # Drop the oldest backup if at capacity
        oldest = os.path.join(_DATA_DIR, f"model_backup_{MAX_BACKUPS}.pkl")
        if os.path.exists(oldest):
            os.remove(oldest)

        # Shift: backup_2 → backup_3, backup_1 → backup_2
        for i in range(MAX_BACKUPS - 1, 0, -1):
            src = os.path.join(_DATA_DIR, f"model_backup_{i}.pkl")
            dst = os.path.join(_DATA_DIR, f"model_backup_{i + 1}.pkl")
            if os.path.exists(src):
                shutil.move(src, dst)

        # Current model → backup_1
        shutil.copy2(model_path, os.path.join(_DATA_DIR, "model_backup_1.pkl"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _append_log(self, result: dict, counts: dict) -> None:
        """Append one entry to retrain_log.json (creates file if absent)."""
        os.makedirs(_DATA_DIR, exist_ok=True)

        entry = {
            "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
            "algorithm":  result.get("algorithm"),
            "n_samples":  result.get("n_samples"),
            "val_f1":     result.get("val_f1"),
            "normal":     counts["normal"],
            "tremor":     counts["tremor"],
        }

        log = []
        if os.path.exists(_LOG_PATH):
            try:
                with open(_LOG_PATH) as f:
                    log = json.load(f)
            except Exception:
                log = []

        log.append(entry)

        with open(_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _days_since_last_train(self) -> float:
        """Days elapsed since the last successful training run."""
        with self._lock:
            if self._last_train_time is None:
                # Use a large number to trigger time-based retrain on very
                # first run if there is already enough data (e.g. after restart)
                return float(RETRAIN_EVERY_DAYS)
            delta = datetime.datetime.utcnow() - self._last_train_time
            return delta.total_seconds() / 86400.0

"""
ml/buffer.py

Ring buffer and window extractor for streaming IMU data.

Responsibilities:
- Accept one IMU frame at a time (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
- Maintain a rolling buffer of the last BUFFER_MAXLEN samples
- Every STEP_SIZE new samples, expose a WINDOW_SIZE-sample window for feature extraction
- Thread-safe: called from websocket_reader thread, read from ml_pipeline thread

Window strategy:
  WINDOW_SIZE = 100 samples  (2 seconds @ 50 Hz)
  STEP_SIZE   = 50  samples  (1 second — 50% overlap)
  BUFFER_MAXLEN = 300 samples (6 seconds — always has room for a full window)
"""

import threading
import numpy as np
from collections import deque

WINDOW_SIZE   = 100   # samples in one analysis window (2s @ 50Hz)
STEP_SIZE     = 50    # new samples needed before next window is ready (1s)
BUFFER_MAXLEN = 300   # maximum samples kept in the ring buffer


class RingBuffer:
    """
    Thread-safe ring buffer for 6-axis IMU data.

    Each entry is a 6-element tuple:
        (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)

    Physical units expected (already converted by process_sensor_line):
        ax_g   : accelerometer X in g
        ay_g   : accelerometer Y in g
        az_g   : accelerometer Z in g
        gx_dps : gyroscope X in degrees/second
        gy_dps : gyroscope Y in degrees/second
        gz_dps : gyroscope Z in degrees/second
    """

    def __init__(self,
                 maxlen: int = BUFFER_MAXLEN,
                 window_size: int = WINDOW_SIZE,
                 step_size: int = STEP_SIZE):
        self._buf        = deque(maxlen=maxlen)
        self._window_size = window_size
        self._step_size   = step_size
        self._new_count   = 0          # samples added since last window was emitted
        self._lock        = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_sample(self,
                   ax_g: float, ay_g: float, az_g: float,
                   gx_dps: float, gy_dps: float, gz_dps: float) -> None:
        """
        Append one IMU frame.  O(1), safe to call from the WebSocket thread.
        """
        with self._lock:
            self._buf.append((ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps))
            self._new_count += 1

    def get_window(self) -> "np.ndarray | None":
        """
        Return a window ready for feature extraction, or None if not ready yet.

        A window is ready when:
          1. Buffer holds at least WINDOW_SIZE samples, AND
          2. At least STEP_SIZE new samples have arrived since the last call
             that returned a window.

        Returns
        -------
        np.ndarray of shape (WINDOW_SIZE, 6)  — columns: ax, ay, az, gx, gy, gz
        None                                  — window not ready yet
        """
        with self._lock:
            if len(self._buf) < self._window_size:
                return None
            if self._new_count < self._step_size:
                return None

            # Take the most recent WINDOW_SIZE samples
            window = np.array(list(self._buf)[-self._window_size:], dtype=np.float32)
            self._new_count = 0   # reset step counter
            return window

    # ------------------------------------------------------------------
    # Diagnostics (useful for logging / unit tests)
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Current number of samples in the buffer."""
        with self._lock:
            return len(self._buf)

    @property
    def samples_since_last_window(self) -> int:
        """How many new samples have arrived since the last emitted window."""
        with self._lock:
            return self._new_count

    def is_warm(self) -> bool:
        """True once the buffer contains a full window's worth of data."""
        with self._lock:
            return len(self._buf) >= self._window_size

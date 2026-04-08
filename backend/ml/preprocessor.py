"""
ml/preprocessor.py

Preprocessing pipeline for a single IMU window.

Input  : np.ndarray of shape (100, 6)  — raw physical units
           columns: ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
Output : np.ndarray of shape (100, 6)  — cleaned signal, same layout

Steps applied in order:
  1. Spike clipping      — clamp samples beyond ±4σ per axis
  2. Low-pass filter     — Butterworth 4th-order, 20 Hz cutoff
                           removes electrical noise above tremor band
                           (Nyquist = 25 Hz at 50 Hz sampling rate)
  3. Gravity removal     — subtract per-window mean from accel axes only
                           isolates dynamic acceleration; leaves gyro unchanged

Why these choices:
  - sosfilt (second-order sections) is numerically stable for real-time windows;
    avoids the numerical blow-up that ba-form lfilter can produce on short data.
  - Spike clipping BEFORE filtering prevents a single bad sample from
    ringing through the Butterworth impulse response.
  - Mean subtraction for gravity is sufficient for a 2-second window because
    the DC component of gravity is constant over that duration.
"""

import numpy as np
from scipy.signal import butter, sosfilt

# Column indices
AX, AY, AZ = 0, 1, 2
GX, GY, GZ = 3, 4, 5
ACCEL_COLS = [AX, AY, AZ]

# Filter design — computed once at import time (static, never changes)
_SAMPLING_RATE   = 50.0   # Hz
_LOWPASS_CUTOFF  = 20.0   # Hz  — keeps tremor band (4-6 Hz) and removes noise
_FILTER_ORDER    = 4
_SPIKE_SIGMA     = 4.0    # clip beyond ±4 standard deviations

_sos = butter(
    _FILTER_ORDER,
    _LOWPASS_CUTOFF / (_SAMPLING_RATE / 2.0),   # normalised cutoff (Wn)
    btype='low',
    output='sos'
)


def preprocess(window: np.ndarray) -> np.ndarray:
    """
    Clean one IMU window.

    Parameters
    ----------
    window : np.ndarray, shape (N, 6), dtype float32 or float64
        Raw IMU samples in physical units.
        Columns: ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps

    Returns
    -------
    np.ndarray, shape (N, 6), dtype float32
        Preprocessed window — spike-free, low-pass filtered, gravity removed.
    """
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) window, got {window.shape}")

    out = window.astype(np.float32, copy=True)

    # ------------------------------------------------------------------
    # Step 1 — Spike clipping (per axis)
    # ------------------------------------------------------------------
    for col in range(6):
        col_data = out[:, col]
        mu  = np.mean(col_data)
        std = np.std(col_data)
        if std > 0:
            lo = mu - _SPIKE_SIGMA * std
            hi = mu + _SPIKE_SIGMA * std
            out[:, col] = np.clip(col_data, lo, hi)

    # ------------------------------------------------------------------
    # Step 2 — Low-pass filter (all 6 axes)
    # ------------------------------------------------------------------
    for col in range(6):
        out[:, col] = sosfilt(_sos, out[:, col])

    # ------------------------------------------------------------------
    # Step 3 — Gravity removal (accelerometer axes only)
    # ------------------------------------------------------------------
    for col in ACCEL_COLS:
        out[:, col] -= np.mean(out[:, col])

    return out

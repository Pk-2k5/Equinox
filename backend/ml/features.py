"""
ml/features.py

Extract a fixed 28-element feature vector from one preprocessed IMU window.

Input  : np.ndarray shape (N, 6) — cleaned window from preprocessor.py
           columns: ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
Output : np.ndarray shape (28,)  — float32 feature vector

Feature layout (in order):
  [0–5]   rms per axis             (ax, ay, az, gx, gy, gz)
  [6–11]  variance per axis        (ax, ay, az, gx, gy, gz)
  [12–17] tremor band power 4–6 Hz (ax, ay, az, gx, gy, gz)
  [18–23] peak frequency 1–25 Hz   (ax, ay, az, gx, gy, gz)
  [24]    resultant accelerometer RMS   √(ax²+ay²+az²)
  [25]    resultant gyroscope RMS       √(gx²+gy²+gz²)
  [26]    spectral entropy of resultant accelerometer
  [27]    zero-crossing rate of resultant accelerometer

All features are reproducible given the same window — no randomness, no state.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq

_SAMPLING_RATE   = 50.0   # Hz
_TREMOR_LO       = 4.0    # Hz  — lower bound of Parkinson's tremor band
_TREMOR_HI       = 6.0    # Hz  — upper bound of Parkinson's tremor band
_PEAK_FREQ_LO    = 1.0    # Hz  — ignore DC when searching for peak frequency
_EPSILON         = 1e-10  # prevents log(0) in entropy

# Human-readable names — same index order as the feature vector
FEATURE_NAMES = [
    # RMS per axis
    "rms_ax", "rms_ay", "rms_az", "rms_gx", "rms_gy", "rms_gz",
    # Variance per axis
    "var_ax", "var_ay", "var_az", "var_gx", "var_gy", "var_gz",
    # Tremor band power (4–6 Hz) per axis
    "tbp_ax", "tbp_ay", "tbp_az", "tbp_gx", "tbp_gy", "tbp_gz",
    # Peak frequency per axis
    "pf_ax",  "pf_ay",  "pf_az",  "pf_gx",  "pf_gy",  "pf_gz",
    # Combined
    "resultant_acc_rms",
    "resultant_gyro_rms",
    "spectral_entropy_acc",
    "zero_cross_rate_acc",
]

assert len(FEATURE_NAMES) == 28


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(window: np.ndarray) -> np.ndarray:
    """
    Compute the 28-element feature vector for one window.

    Parameters
    ----------
    window : np.ndarray, shape (N, 6)
        Preprocessed IMU window. N is typically 100 (2 s @ 50 Hz).

    Returns
    -------
    np.ndarray, shape (28,), dtype float32
    """
    if window.ndim != 2 or window.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) window, got {window.shape}")

    n = window.shape[0]
    freqs = rfftfreq(n, d=1.0 / _SAMPLING_RATE)   # frequency axis for FFT bins

    # Pre-compute FFT magnitudes for all 6 axes at once (n//2+1 bins each)
    fft_mag = np.abs(rfft(window, axis=0))          # shape: (n//2+1, 6)

    features = np.empty(28, dtype=np.float32)

    # ------------------------------------------------------------------
    # Features 0–5 : RMS per axis
    # ------------------------------------------------------------------
    features[0:6] = np.sqrt(np.mean(window ** 2, axis=0))

    # ------------------------------------------------------------------
    # Features 6–11 : Variance per axis
    # ------------------------------------------------------------------
    features[6:12] = np.var(window, axis=0)

    # ------------------------------------------------------------------
    # Features 12–17 : Tremor band power (4–6 Hz) per axis
    # Computed as fraction of total power in that axis so that the value
    # is comparable across sessions regardless of sensor orientation.
    # ------------------------------------------------------------------
    tremor_mask = (freqs >= _TREMOR_LO) & (freqs <= _TREMOR_HI)
    total_power = np.sum(fft_mag ** 2, axis=0) + _EPSILON   # shape: (6,)
    tremor_power = np.sum(fft_mag[tremor_mask] ** 2, axis=0) # shape: (6,)
    features[12:18] = (tremor_power / total_power).astype(np.float32)

    # ------------------------------------------------------------------
    # Features 18–23 : Peak frequency per axis (1–25 Hz, ignores DC)
    # ------------------------------------------------------------------
    peak_mask = freqs >= _PEAK_FREQ_LO
    masked_fft = fft_mag.copy()
    masked_fft[~peak_mask] = 0.0
    peak_bin = np.argmax(masked_fft, axis=0)               # shape: (6,)
    features[18:24] = freqs[peak_bin].astype(np.float32)

    # ------------------------------------------------------------------
    # Feature 24 : Resultant accelerometer RMS
    # RMS of the vector magnitude √(ax²+ay²+az²) — captures overall
    # movement intensity regardless of wrist orientation.
    # ------------------------------------------------------------------
    resultant_acc = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
    features[24] = np.sqrt(np.mean(resultant_acc ** 2))

    # ------------------------------------------------------------------
    # Feature 25 : Resultant gyroscope RMS
    # ------------------------------------------------------------------
    resultant_gyro = np.sqrt(window[:, 3]**2 + window[:, 4]**2 + window[:, 5]**2)
    features[25] = np.sqrt(np.mean(resultant_gyro ** 2))

    # ------------------------------------------------------------------
    # Feature 26 : Spectral entropy of resultant accelerometer
    # High entropy → broadband noise (walking, vibration)
    # Low entropy  → narrowband, periodic signal (tremor)
    # ------------------------------------------------------------------
    acc_fft_mag = np.abs(rfft(resultant_acc))
    acc_power   = acc_fft_mag ** 2
    acc_power_norm = acc_power / (np.sum(acc_power) + _EPSILON)
    features[26] = float(-np.sum(acc_power_norm * np.log(acc_power_norm + _EPSILON)))

    # ------------------------------------------------------------------
    # Feature 27 : Zero-crossing rate of resultant accelerometer
    # Tremor has regular crossings; walking has irregular, higher rate.
    # ------------------------------------------------------------------
    signs = np.sign(resultant_acc - np.mean(resultant_acc))
    crossings = np.sum(np.diff(signs) != 0)
    features[27] = float(crossings) / float(n - 1)

    return features

"""Noise injection at a target SNR.

    SNR_dB = 20 * log10( rms(x - mean(x)) / rms(noise) )

Not the definition in Pal et al. 2024, which uses the RMS of the IEM filter's
non-stationary output, so curves here aren't numerically comparable to theirs.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def add_noise_at_snr(x: np.ndarray, snr_db: float, fs: float = 256.0,
                     band: tuple[float, float] | None = None,
                     seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise scaled to hit `snr_db` exactly.

    `band` optionally restricts the noise to a frequency band, e.g. (0.5, 40),
    which is more representative of physiological interference than white noise.
    """
    rng = np.random.default_rng(seed)
    ac = x - np.mean(x)
    noise = rng.standard_normal(len(x))
    if band is not None:
        lo, hi = band
        b, a = butter(4, [lo / (fs / 2), min(hi, fs / 2 - 1e-6) / (fs / 2)], btype="band")
        noise = filtfilt(b, a, noise)
    target = _rms(ac) / (10 ** (snr_db / 20.0))
    cur = _rms(noise)
    if cur == 0:
        return x.copy()
    return x + noise * (target / cur)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Verify the achieved SNR, for use in tests and in the results table."""
    ac = clean - np.mean(clean)
    return 20.0 * np.log10(_rms(ac) / max(_rms(noisy - clean), 1e-12))

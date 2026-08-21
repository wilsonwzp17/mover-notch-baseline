"""Systolic peaks and second-derivative dicrotic-notch detection.

The comparison baseline from Pal et al. 2024, not their IEM method.
Preprocessing follows the paper: 4th-order Butterworth low-pass at 16 Hz, then
Savitzky-Golay smoothing before differentiating.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


def lowpass(x: np.ndarray, fs: float = 256.0, cutoff: float = 16.0,
            order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def find_systolic_peaks(x: np.ndarray, fs: float = 256.0,
                        min_hr: float = 40.0, max_hr: float = 180.0) -> np.ndarray:
    """Systolic peaks with a physiological refractory distance."""
    min_dist = int(fs * 60.0 / max_hr)
    prom = 0.3 * np.std(x)
    peaks, _ = find_peaks(x, distance=max(1, min_dist), prominence=max(prom, 1e-9))
    return peaks


def second_derivative(x: np.ndarray, fs: float = 256.0,
                      half_width_s: float = 0.1, polyorder: int = 4) -> np.ndarray:
    """Savitzky-Golay second derivative.

    Window length is tied to the sampling rate rather than hard-coded, so the
    detector behaves consistently if MOVER's rate differs from MLORD's 256 Hz.
    """
    w = int(round(half_width_s * fs)) | 1  # force odd
    w = max(w, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if w % 2 == 0:
        w += 1
    return savgol_filter(x, window_length=w, polyorder=polyorder, deriv=2, delta=1.0 / fs)


def detect_notch_2nd_deriv(x: np.ndarray, fs: float = 256.0,
                           peaks: np.ndarray | None = None,
                           min_delay_s: float = 0.1,
                           max_frac_of_beat: float = 0.85) -> list[int | None]:
    """Per-beat dicrotic-notch index, or None where no candidate is found.

    The notch appears as a local maximum of the second derivative occurring at
    least `min_delay_s` after the systolic peak, consistent with the published
    constraint that the notch sits at least 0.1 s from the systolic peak.
    """
    xf = lowpass(x, fs)
    if peaks is None:
        peaks = find_systolic_peaks(xf, fs)
    d2 = second_derivative(xf, fs)

    out: list[int | None] = []
    for i, p in enumerate(peaks):
        nxt = peaks[i + 1] if i + 1 < len(peaks) else len(xf) - 1
        lo = p + int(min_delay_s * fs)
        hi = p + int((nxt - p) * max_frac_of_beat)
        if hi <= lo or hi >= len(d2):
            out.append(None)
            continue
        seg = d2[lo:hi]
        cand, props = find_peaks(seg, prominence=0)
        if not len(cand):
            out.append(None)
            continue
        # Most prominent d2 max, not the first one. The first maximum lands on
        # an early shoulder of the decay limb (133 ms vs 241 ms here).
        out.append(int(lo + cand[int(np.argmax(props["prominences"]))]))
    return out


def find_systolic_onsets(x: np.ndarray, fs: float,
                         peaks: np.ndarray | None = None,
                         max_upstroke_s: float = 0.4) -> list[int | None]:
    """Systolic onset: the pressure minimum before each peak.

    Needed for onset-to-notch, which is what Pal et al. call systolic phase
    duration. Returns None if nothing is found in the window.
    """
    xf = lowpass(x, fs)
    if peaks is None:
        peaks = find_systolic_peaks(xf, fs)
    back = int(max_upstroke_s * fs)
    out: list[int | None] = []
    for p in peaks:
        a = max(0, p - back)
        seg = xf[a:p]
        if len(seg) < 3:
            out.append(None)
            continue
        mins, _ = find_peaks(-seg)
        out.append(int(a + mins[-1]) if len(mins) else int(a + int(np.argmin(seg))))
    return out

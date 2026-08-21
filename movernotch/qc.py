"""Window screening. Every rule is explicit and reports why a window was dropped,
so exclusions can be counted per SNR level instead of disappearing silently.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class QCResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def screen_window(x: np.ndarray, fs: float = 256.0, *,
                  abp_mmhg: bool = False,
                  min_beats: int = 2,
                  flat_tol: float = 1e-6,
                  max_peaks_per_sec: float = 3.0) -> QCResult:
    r: list[str] = []
    if np.any(~np.isfinite(x)):
        r.append("non_finite_samples")
    if np.std(x) < flat_tol:
        r.append("flatline")
    dur = len(x) / fs
    from .detect import find_systolic_peaks, lowpass
    try:
        peaks = find_systolic_peaks(lowpass(x, fs), fs)
    except Exception:
        peaks = np.array([], dtype=int)
    if len(peaks) < min_beats:
        r.append("too_few_beats")
    if dur > 0 and len(peaks) / dur > max_peaks_per_sec:
        r.append("implausible_beat_rate")
    if abp_mmhg:
        # Physiologic-plausibility gate, which also surfaces the waveform gain
        # errors MOVER warns about in its non-v2 archives.
        if np.nanmin(x) < 10 or np.nanmax(x) > 300:
            r.append("abp_out_of_physiologic_range")
        if np.nanmax(x) - np.nanmin(x) < 5:
            r.append("abp_pulse_pressure_implausible")
    return QCResult(ok=len(r) == 0, reasons=r)

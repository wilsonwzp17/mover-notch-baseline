#!/usr/bin/env python3
"""Landmark detection on one real record. Source of the numbers in the READMEs.

    python experiments/validate_record.py [path-to.xml]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import io as mio, detect
from scipy.signal import find_peaks as _find_peaks  # noqa: E402

WINDOW_S = 4.0


def window_passes(w: np.ndarray) -> bool:
    """Window gate; record-level screening happens in scan_usable.py."""
    s = w[np.isfinite(w)]
    if s.size < len(w) * 0.9:
        return False
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return bool(20 <= lo and hi <= 300 and (hi - lo) >= 15)


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        listing = Path("data/usable_records.txt")
        if not listing.exists():
            raise SystemExit("run scripts/scan_usable.py first, or pass a record path")
        path = Path(listing.read_text().splitlines()[0].split("\t")[0])

    rec = mio.load_archive(path).channels["GE_ART"]
    fs = rec.fs
    n_win = n_ok = n_peaks = n_notches = 0
    hr, p2n = [], []

    for _, w in mio.iter_windows(rec.signal, fs, WINDOW_S):
        n_win += 1
        if window_passes(w):
            n_ok += 1

    # Whole-record analysis, so beats are not clipped at window edges.
    x = np.nan_to_num(rec.signal, nan=float(np.nanmedian(rec.signal)))
    xf = detect.lowpass(x, fs)
    peaks = detect.find_systolic_peaks(xf, fs)
    notches = detect.detect_notch_2nd_deriv(x, fs=fs, peaks=peaks)
    onsets = detect.find_systolic_onsets(x, fs, peaks=peaks)
    n_peaks = len(peaks)
    n_notches = sum(n is not None for n in notches)

    p2n_arr = np.array([(n - p) / fs * 1000.0
                        for p, n in zip(peaks, notches) if n is not None])
    o2n_arr = np.array([(n - o) / fs * 1000.0
                        for o, n in zip(onsets, notches)
                        if o is not None and n is not None])

    # Independent reference: the incisura is the first pressure minimum after the
    # same 0.1 s delay used by the detector, inside the same diastolic window.
    offs = []
    for i in range(len(peaks) - 1):
        nt = notches[i]
        if nt is None:
            continue
        lo = peaks[i] + int(0.1 * fs)
        hi = peaks[i] + int((peaks[i + 1] - peaks[i]) * 0.85)
        if hi <= lo or hi >= len(xf):
            continue
        m, _ = _find_peaks(-xf[lo:hi])
        if len(m):
            offs.append((nt - (lo + m[0])) / fs * 1000.0)
    offs = np.array(offs)

    print(f"record                     {path.name}")
    print(f"channel                    GE_ART @ {fs:.1f} Hz measured "
          f"(declared {rec.fs_declared:.0f}), {len(rec.signal)/fs/60:.1f} min")
    print(f"windows ({WINDOW_S:.0f} s)             {n_win}")
    print(f"windows passing QC         {n_ok}  ({100*n_ok/max(n_win,1):.1f}%)")
    print(f"systolic peaks             {n_peaks}")
    print(f"dicrotic notches           {n_notches}  "
          f"({100*n_notches/max(n_peaks,1):.1f}% of beats)")
    print(f"heart rate (beats/span)    {n_peaks/(len(rec.signal)/fs)*60:.1f} bpm")
    print(f"median peak-to-notch       {np.median(p2n_arr):.1f} ms  "
          f"(IQR {np.percentile(p2n_arr,25):.0f} to {np.percentile(p2n_arr,75):.0f})")
    print(f"median onset-to-notch      {np.median(o2n_arr):.1f} ms  "
          f"(IQR {np.percentile(o2n_arr,25):.0f} to {np.percentile(o2n_arr,75):.0f})")
    print(f"vs incisura reference      median {np.median(offs):+.1f} ms, within 30 ms "
          f"on {(np.abs(offs)<=30).mean()*100:.1f}% of {len(offs)} beats")
    print()
    print("NOTE: peak-to-notch is measured from the SYSTOLIC PEAK. It is not the")
    print("systolic phase duration of Pal et al., which is measured from systolic")
    print("ONSET. The two are not directly comparable.")


if __name__ == "__main__":
    main()

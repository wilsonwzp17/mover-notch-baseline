#!/usr/bin/env python3
"""Overlay detected landmarks on one real MOVER arterial record.

    python experiments/plot_record.py            # writes results/real_data_landmarks.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import io as mio, detect  # noqa: E402
from experiments.validate_record import window_passes  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    Path("data/usable_records.txt").read_text().splitlines()[0].split("\t")[0])
rec = mio.load_archive(path).channels["GE_ART"]
fs = rec.fs

win = None
for _, c in mio.iter_windows(rec.signal, fs, 8.0):
    if window_passes(c):
        win = c
        break
xf = detect.lowpass(win, fs)
peaks = detect.find_systolic_peaks(xf, fs)
notches = detect.detect_notch_2nd_deriv(win, fs=fs, peaks=peaks)
t = np.arange(len(win)) / fs

p2n = []
for _, w in mio.iter_windows(rec.signal, fs, 4.0):
    if not window_passes(w):
        continue
    wf = np.nan_to_num(w, nan=float(np.nanmedian(w)))
    pk = detect.find_systolic_peaks(detect.lowpass(wf, fs), fs)
    for p, n in zip(pk, detect.detect_notch_2nd_deriv(wf, fs=fs, peaks=pk)):
        if n is not None:
            p2n.append((n - p) / fs * 1000.0)

fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
ax[0].plot(t, win, lw=.8, color="0.6", label="raw")
ax[0].plot(t, xf, lw=1.4, color="C0", label="low-pass 16 Hz")
ax[0].plot(t[peaks], xf[peaks], "^", ms=8, color="C2", label="systolic peak")
ok = [n for n in notches if n is not None]
ax[0].plot(t[ok], xf[ok], "v", ms=8, color="C3",
           label="dicrotic notch (second-derivative baseline)")
ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("arterial pressure (mmHg)")
ax[0].set_title(f"MOVER IP record A, GE_ART @ {fs:.1f} Hz measured (declared 180), v2, XML gain 0.25")
ax[0].legend(loc="upper right", fontsize=8)
ax[0].grid(alpha=.3)

ax[1].hist(p2n, bins=40, color="C0", alpha=.85)
ax[1].set_xlabel("peak-to-notch interval, systolic PEAK to detected notch (ms)")
ax[1].set_ylabel("beats")
ax[1].grid(alpha=.3)
ax[1].set_title(
    f"n={len(p2n)} beats, median {np.median(p2n):.0f} ms. Measured from the systolic "
    f"peak, so NOT comparable to\nthe systolic phase duration of Pal et al., which is "
    f"measured from systolic onset.", fontsize=8.5)
fig.tight_layout()
out = Path("results/real_data_landmarks.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}  (n={len(p2n)} beats, median {np.median(p2n):.1f} ms)")

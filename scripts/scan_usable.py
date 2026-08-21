#!/usr/bin/env python3
"""Inventory which MOVER records actually carry usable arterial or PPG waveforms.

A channel being present does not mean a transducer was connected. This scans
assembled archives and applies physiologic screens, so the study is built on a
known denominator instead of on whatever the first file happened to contain.
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import io as mio

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample")
pattern = sys.argv[2] if len(sys.argv) > 2 else "*IP-*.xml"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 150

files = sorted(root.rglob(pattern))
step = max(1, len(files) // limit)
files = files[::step][:limit]
print(f"scanning {len(files)} of {len(sorted(root.rglob(pattern)))} files matching {pattern}\n")

def verdict(ch: str, sig: np.ndarray) -> tuple[str, dict]:
    s = sig[np.isfinite(sig)]
    if s.size < 100:
        return "empty", {}
    lo, hi = float(np.percentile(s, 1)), float(np.percentile(s, 99))
    pp = hi - lo
    st = {"p1": lo, "p99": hi, "pulse": pp, "std": float(np.std(s))}
    if ch.upper() in ("GE_ART", "INVP1", "ABP"):
        if pp < 5:                      return "flat_no_transducer", st
        if lo < -20 or hi > 300:        return "out_of_range", st
        if pp < 15:                     return "damped_or_weak", st
        if 20 <= lo and hi <= 300:      return "USABLE_ABP", st
        return "questionable", st
    if "PLETH" in ch.upper():
        return ("USABLE_PPG" if pp > 5 else "flat_no_transducer"), st
    return "other", st

tally = collections.Counter()
usable, per_case = [], collections.defaultdict(set)
for f in files:
    try:
        a = mio.load_archive(f)
    except Exception as e:
        tally[f"parse_error:{type(e).__name__}"] += 1
        continue
    for ch, rec in a.channels.items():
        v, st = verdict(ch, rec.signal)
        tally[f"{ch}:{v}"] += 1
        if v.startswith("USABLE"):
            usable.append((f, ch, rec.fs, len(rec.signal) / rec.fs / 60, st))
            per_case[f.name.split("IP-")[0].split("CB-")[0]].add(ch)

print("=== verdicts ===")
for k, n in sorted(tally.items(), key=lambda x: -x[1]):
    print(f"  {n:4d}  {k}")

print(f"\n=== usable records: {len(usable)} across {len(per_case)} cases ===")
for f, ch, fs, mins, st in usable[:15]:
    print(f"  {f.name[:40]:42} {ch:8} {fs:5.0f}Hz {mins:5.1f}min  "
          f"p1={st['p1']:6.1f} p99={st['p99']:6.1f} pulse={st['pulse']:5.1f}")
if usable:
    out = Path("data/usable_records.txt")
    out.write_text("\n".join(f"{f}\t{ch}\t{fs}" for f, ch, fs, _, _ in usable))
    print(f"\nwrote {out}")

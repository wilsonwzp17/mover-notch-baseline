#!/usr/bin/env python3
"""Characterize a sampled set of MOVER waveform XML files.

Answers the questions that must be settled before any analysis:
  what channels exist, at what sampling rate, what gain/offset convention,
  what the CB vs IP file types actually contain, and whether the v2 data still
  needs the official gain overrides.

    python scripts/inspect_sample.py data/raw/sample
"""
from __future__ import annotations
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from movernotch import io as mio

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 150
all_files = sorted(root.rglob("*.xml"))
# Sample evenly across the tree so both file types and many cases are covered.
step = max(1, len(all_files) // LIMIT)
files = all_files[::step][:LIMIT]
print(f"(inspecting {len(files)} of {len(all_files)} files)")
if not files:
    raise SystemExit(f"no XML under {root}")

def ftype(p: Path) -> str:
    stem = p.name
    for tag in ("IP-", "CB-"):
        if tag in stem:
            return tag[:-1]
    return "?"

def caseid(p: Path) -> str:
    n = p.name
    for tag in ("IP-", "CB-"):
        if tag in n:
            return n.split(tag)[0]
    return "?"

print(f"{len(files)} XML files, {len(set(caseid(f) for f in files))} distinct case ids")
print("by type:", dict(collections.Counter(ftype(f) for f in files)))
print()

chan_stats: dict[tuple[str, str], list] = collections.defaultdict(list)
errors = 0
for f in files:
    try:
        recs = mio.load_waveform_records(f, apply_gain_overrides=False)
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  parse error {f.name}: {type(e).__name__}: {e}")
        continue
    for r in recs:
        chan_stats[(ftype(f), r.channel)].append(
            (r.fs, len(r.signal), r.gain, r.offset,
             float(np.nanmin(r.signal)), float(np.nanmax(r.signal))))

if errors:
    print(f"\n{errors} files failed to parse\n")

print(f"{'type':5} {'channel':16} {'n':>5} {'Hz':>7} {'samples':>9} {'gain':>8} "
      f"{'offset':>7} {'min':>9} {'max':>9}  duration")
print("-" * 100)
for (t, ch), rows in sorted(chan_stats.items()):
    a = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows], dtype=float)
    fs, n = np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1])
    dur = n / fs if fs else float("nan")
    print(f"{t:5} {ch:16} {len(rows):5d} {fs:7.1f} {n:9.0f} {np.nanmedian(a[:,2]):8.4f} "
          f"{np.nanmedian(a[:,3]):7.2f} {np.nanmin(a[:,4]):9.2f} {np.nanmax(a[:,5]):9.2f}  {dur/60:.1f} min")

# The gain question, resolved empirically on a pressure-like channel.
print("\n=== GAIN CONVENTION CHECK ===")
target = None
for f in files:
    recs = mio.load_waveform_records(f, apply_gain_overrides=False)
    if any(c in r.channel.upper() for r in recs for c in ("ART", "INVP", "ABP")):
        target = f
        break
if target is None:
    print("no arterial-like channel found in this sample; pull more files or a different shard")
else:
    print(f"file: {target.name}")
    for k, v in mio.diagnose_gain_convention(target).items():
        print(f"  {k}: {v}")

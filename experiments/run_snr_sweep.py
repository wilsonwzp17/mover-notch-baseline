#!/usr/bin/env python3
"""Detection rate and timing error vs SNR.

    python experiments/run_snr_sweep.py --config configs/sweep_mover.yaml
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import synth, noise as noisemod, detect, evaluate, qc, io as mio  # noqa: E402


def load_config(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = float(cfg["fs"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    source = cfg.get("source", "synthetic")
    if source == "synthetic":
        clean, sys_idx, notch_idx = synth.synth_record(
            n_beats=int(cfg["n_beats"]), fs=fs, seed=int(cfg["seed"]))
        provenance = f"synthetic fixture, {cfg['n_beats']} beats"
    elif source == "mover":
        # References are landmarks detected at native quality, then degraded, so
        # this measures self-consistency under noise, not agreement with a marker.
        from experiments.validate_record import window_passes
        listing = Path(cfg.get("records_file", "data/usable_records.txt"))
        rows = [l.split("\t") for l in listing.read_text().splitlines() if l.strip()]
        max_rec = int(cfg.get("max_records", 1))
        max_win = int(cfg.get("max_windows", 200))
        per_rec, fs_seen = [], set()
        for row in rows[:max_rec]:
            rec = mio.load_archive(Path(row[0])).channels[row[1]]
            fs_seen.add(round(float(rec.fs), 1))
            keep = [np.nan_to_num(w, nan=float(np.nanmedian(w)))
                    for _, w in mio.iter_windows(rec.signal, rec.fs, 4.0)
                    if window_passes(w)]
            per_rec.append((Path(row[0]).name, keep))
        if len(fs_seen) != 1:
            raise SystemExit(f"records disagree on sampling rate: {fs_seen}")
        fs = fs_seen.pop()

        # Draw evenly from every record rather than filling from the first, which
        # silently made a multi-record run a single-record one.
        n_rec = len(per_rec)
        quota = max(1, max_win // n_rec) if n_rec else 0
        segs, contributing = [], []
        for name, keep in per_rec:
            take = keep[:quota]
            if take:
                contributing.append(name)
            segs.extend(take)
        # Backfill any shortfall from records that still have windows left.
        if len(segs) < max_win:
            for name, keep in per_rec:
                for w in keep[quota:]:
                    if len(segs) >= max_win:
                        break
                    segs.append(w)
                    if name not in contributing:
                        contributing.append(name)
                if len(segs) >= max_win:
                    break
        segs = segs[:max_win]
        clean = np.concatenate(segs)
        peaks_ref = detect.find_systolic_peaks(detect.lowpass(clean, fs), fs)
        ref = detect.detect_notch_2nd_deriv(clean, fs=fs, peaks=peaks_ref)
        notch_idx = np.array([n for n in ref if n is not None])
        sys_idx = peaks_ref
        provenance = (f"MOVER, {len(contributing)} record(s) contributing, {len(segs)} "
                      f"QC-passing 4 s windows, {len(notch_idx)} reference notches "
                      f"@ {fs:.1f} Hz (measured)")
    else:
        raise SystemExit(f"unknown source: {source}")
    print(f"source: {provenance}\n")

    rows = []
    snrs = np.arange(cfg["snr_db_min"], cfg["snr_db_max"] + 1, cfg["snr_db_step"])
    for snr in snrs:
        noisy = noisemod.add_noise_at_snr(
            clean, float(snr), fs=fs,
            band=tuple(cfg["noise_band"]) if cfg.get("noise_band") else None,
            seed=int(cfg["seed"]))
        achieved = noisemod.measured_snr_db(clean, noisy)
        scr = qc.screen_window(noisy, fs, abp_mmhg=bool(cfg.get("abp_mmhg", False)))
        peaks = detect.find_systolic_peaks(detect.lowpass(noisy, fs), fs)
        pred = detect.detect_notch_2nd_deriv(noisy, fs=fs, peaks=peaks)

        # Full lists: truncating to min(len) drops the tail of the record.
        summ = evaluate.summarize(pred, notch_idx, fs=fs)
        errs, _ = evaluate.match_errors(pred, notch_idx, fs=fs)
        lo, hi = evaluate.bootstrap_ci(np.abs(errs)) if len(errs) else (float("nan"),) * 2

        rows.append({"snr_db_target": float(snr), "snr_db_achieved": round(achieved, 3),
                     "qc_ok": scr.ok, "qc_reasons": "|".join(scr.reasons),
                     "n_beats_reference": int(len(notch_idx)), "n_peaks_found": int(len(peaks)),
                     "abs_err_ci_lo_s": lo, "abs_err_ci_hi_s": hi, **summ})
        print(f"SNR {snr:>4} dB  det={summ['detection_rate']:.3f}  "
              f"|err|={summ['mean_abs_error_s']*1000 if summ['mean_abs_error_s']==summ['mean_abs_error_s'] else float('nan'):.1f} ms  "
              f"<=30ms={summ['success_within_30ms']:.3f}")

    csv_path = out / ("snr_sweep_mover.csv" if source == "mover" else "snr_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "run_config.json").write_text(json.dumps({**cfg, "provenance": provenance}, indent=2))
    print(f"\nwrote {csv_path}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["snr_db_target"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(x, [r["detection_rate"] for r in rows], marker="o", ms=3)
        ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("detection rate"); ax[0].set_ylim(0, 1.02)
        ax[1].plot(x, [r["success_within_30ms"] for r in rows], marker="o", ms=3, label="<=30 ms")
        ax[1].plot(x, [r["success_within_70ms"] for r in rows], marker="s", ms=3, label="<=70 ms")
        ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("fraction within tolerance"); ax[1].legend()
        for a in ax: a.grid(alpha=.3)
        fig.suptitle(f"Second-derivative dicrotic-notch baseline, recovery vs SNR\n{provenance}", fontsize=9)
        fig.tight_layout(); fig.savefig(out / ("snr_sweep_mover.png" if source == "mover" else "snr_sweep.png"), dpi=150)
        print('wrote figure')
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()

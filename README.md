# mover-notch-baseline

second-derivative dicrotic notch detection on MOVER (public OR waveforms, UCI).
mostly this turned into a thing about the sampling rate being wrong.

## rate

IP files say `Hz=180`. it's ~120. anything you compute in seconds off the header
is 1.5x out.

216000 samples / 1798.6 s of packet timestamps = 120.0. median 119.9 across 45
records from 45 patients, none within 1% of 180. packets are 180 pts every ~1.5 s
not 1.0 (0 of 1200 gaps anywhere near 1 s). `duration="30"` matches the measured
29.98 min; at 180 the file would be 20. CB files in the same archive declare
300/100/100 and the same code recovers those to 0.06% median error so it isn't the
estimator. 2215 beats / 1798.6 s = 73.8 bpm, vs 110.8 if you believe the header.

outside check: onset->notch here is 350.0 ms, Pal 2024 report 344.8 on MLORD.
at 180 you'd get 233.

so `load_archive` measures fs off the timestamps and uses that, declared stays in
`fs_declared`.

## run

    pip install -e ".[dev]"
    pytest -q
    python experiments/validate_record.py
    python experiments/run_snr_sweep.py --config configs/sweep_mover.yaml

needs MOVER access (free, sign their DUA). no data in here.

## one record

    windows 450, QC pass 442 (98.2%)
    peaks 2215, notches 2215
    73.8 bpm
    peak->notch 241.7 ms   onset->notch 350.0 ms

peak->notch is measured from the peak, Pal's SPD is from onset, not the same
number. notch = most prominent d2 max in the diastolic window. against the first
pressure min (incisura): 81.4% within 30 ms over 2185 beats, median -25 ms.
taking the *first* d2 max instead gives 133 ms, way too early.

## sweep

200 QC windows, 2 records, 940 refs.

    SNR   recall  prec    F1     <=30ms  MAE
    -30   0.676   0.377   0.484  0.220   73.7
    -10   0.753   0.409   0.531  0.250   67.3
      0   0.868   0.460   0.602  0.338   55.0
    +10   0.794   0.475   0.594  0.456   47.8
    +20   0.729   0.720   0.724  0.749   28.5
    +30   0.926   0.926   0.926  0.923    9.1
    +40   0.976   0.976   0.976  0.975    3.1

recall isn't monotonic - matching is 150 ms against an ~800 ms beat so a detector
that fires a lot scores well by luck. that's why precision is in there. F1 and
<=30ms basically are monotone. QC rejects the noised signal below +6 dB anyway so
the negative rows are a regime you'd throw out.

## caveats

2 records, one channel, not a generalization claim. the "references" are just the
landmarks found on the clean signal and then degraded, so this is self-consistency
under noise, not agreement with a human marker. my SNR is defined on the raw AC
signal and Pal's is on the IEM output, so the x-axes aren't measuring the same
thing and these curves don't line up with theirs numerically. 2 of 120 IP records
I scanned had a usable art line.

baseline only, no IEM - that's theirs, and they'll send it if you ask.

## data

see data/README. use the `_v2` archives, the earlier ones have wrong gains. don't
redistribute, do share code if you publish, which is what this is.

## refs

- Pal 2024, Comput Methods Programs Biomed 254:108283 (notch / IEM)
- Pal 2025, npj Cardiovasc Health 2(1):57 (feature tool)
- Samad 2023, JAMIA Open 6(4):ooad084 (MOVER)

## todo

- more records, scan_usable finds them, just never ran it wide
- onset detection is crude (min before the peak), good enough for the SPD check
- PPG lives in the CB files, haven't touched it
- INVP1 gain override, never checked whether it's still needed on v2

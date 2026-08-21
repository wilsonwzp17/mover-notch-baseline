"""Synthetic beats with known landmarks. Lets the pipeline be tested without
MOVER access, and gives a controlled ground truth to check the metrics against.
"""
from __future__ import annotations
import numpy as np


def beat_template(fs: float = 256.0, hr_bpm: float = 70.0,
                  sys_amp: float = 1.0, tidal_amp: float = 0.20,
                  dicrotic_amp: float = 0.28, jitter: float = 0.0,
                  rng: np.random.Generator | None = None):
    """One cardiac cycle as a sum of three Gaussian waves.

    Returns (t, y, landmarks) where landmarks holds the *numerically derived*
    ground-truth times: systolic peak and the dicrotic notch, defined as the
    local minimum between the tidal and dicrotic waves on the noiseless template.
    """
    rng = rng or np.random.default_rng(0)
    period = 60.0 / hr_bpm
    n = int(round(period * fs))
    t = np.arange(n) / fs

    j = (lambda s: s * (1.0 + jitter * rng.standard_normal()))
    # Wave positions chosen so the dicrotic notch is a WELL-FORMED local
    # minimum. This fixture is deliberately clean: its job is to exercise the
    # pipeline, not to imitate the difficulty of real recordings. Difficulty is
    # introduced in a controlled way by movernotch.noise instead.
    mu_s, sd_s = j(0.22 * period), 0.055 * period
    mu_t, sd_t = j(0.38 * period), 0.055 * period
    mu_d, sd_d = j(0.62 * period), 0.070 * period

    g = lambda mu, sd, a: a * np.exp(-((t - mu) ** 2) / (2 * sd ** 2))
    y = g(mu_s, sd_s, sys_amp) + g(mu_t, sd_t, tidal_amp) + g(mu_d, sd_d, dicrotic_amp)

    # ground truth: systolic peak, then the minimum before the dicrotic wave
    i_sys = int(np.argmax(y))
    i_dic = int(np.argmin(np.abs(t - mu_d)))
    seg = y[i_sys:i_dic + 1]
    i_notch = i_sys + int(np.argmin(seg)) if len(seg) > 2 else i_sys
    return t, y, {"systolic_idx": i_sys, "notch_idx": i_notch,
                  "systolic_t": t[i_sys], "notch_t": t[i_notch]}


def synth_record(n_beats: int = 60, fs: float = 256.0, hr_bpm: float = 70.0,
                 hr_jitter: float = 0.06, shape_jitter: float = 0.05,
                 seed: int = 0):
    """A multi-beat record plus per-beat ground-truth landmark sample indices."""
    rng = np.random.default_rng(seed)
    chunks, sys_idx, notch_idx, offset = [], [], [], 0
    for _ in range(n_beats):
        hr = hr_bpm * (1.0 + hr_jitter * rng.standard_normal())
        _, y, lm = beat_template(fs=fs, hr_bpm=max(40.0, hr),
                                 jitter=shape_jitter, rng=rng)
        chunks.append(y)
        sys_idx.append(offset + lm["systolic_idx"])
        notch_idx.append(offset + lm["notch_idx"])
        offset += len(y)
    return np.concatenate(chunks), np.array(sys_idx), np.array(notch_idx)

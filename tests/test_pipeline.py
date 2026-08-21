import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movernotch import synth, noise as noisemod, detect, evaluate, qc

FS = 256.0


def test_synth_landmarks_ordered():
    t, y, lm = synth.beat_template(fs=FS)
    assert 0 < lm["systolic_idx"] < lm["notch_idx"] < len(y)


def test_noise_hits_requested_snr():
    clean, _, _ = synth.synth_record(n_beats=20, fs=FS, seed=1)
    for target in (-20.0, -10.0, 0.0):
        noisy = noisemod.add_noise_at_snr(clean, target, fs=FS, seed=1)
        assert abs(noisemod.measured_snr_db(clean, noisy) - target) < 0.5


def test_detector_recovers_notch_on_clean_signal():
    """On a clean signal the detector should find nearly every beat and be ST."""
    clean, _, notch_idx = synth.synth_record(n_beats=30, fs=FS, seed=2)
    peaks = detect.find_systolic_peaks(detect.lowpass(clean, FS), FS)
    pred = detect.detect_notch_2nd_deriv(clean, fs=FS, peaks=peaks)
    s = evaluate.summarize(pred, notch_idx, fs=FS)
    assert s["detection_rate"] > 0.8, s
    assert s["scatter_s"] < 0.040, s
    assert s["success_within_70ms_debiased"] > 0.7, s


def test_detection_degrades_with_noise():
    clean, _, notch_idx = synth.synth_record(n_beats=30, fs=FS, seed=3)
    def score(snr):
        sig = clean if snr is None else noisemod.add_noise_at_snr(clean, snr, fs=FS, seed=3)
        pred = detect.detect_notch_2nd_deriv(sig, fs=FS)
        return evaluate.summarize(pred, notch_idx, fs=FS)["detection_rate"]
    assert score(None) >= score(-25.0)


def test_qc_flags_flatline_and_range():
    flat = np.zeros(int(4 * FS))
    assert not qc.screen_window(flat, FS).ok
    assert "flatline" in qc.screen_window(flat, FS).reasons
    clean, _, _ = synth.synth_record(n_beats=6, fs=FS, seed=4)
    bad_gain = clean * 1000.0
    assert "abp_out_of_physiologic_range" in qc.screen_window(bad_gain, FS, abp_mmhg=True).reasons


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(0.01, 0.002, 200)
    lo, hi = evaluate.bootstrap_ci(np.abs(v))
    assert lo < np.mean(np.abs(v)) < hi


def test_matching_is_robust_to_a_missed_beat():
    """Positional matching would corrupt every beat after a miss; proximity
    matching must not."""
    true = np.array([100, 300, 500, 700])
    pred = [300, 500, 700]          # first beat missed entirely
    s = evaluate.summarize(pred, true, fs=FS)
    assert s["detection_rate"] == 0.75
    assert s["mean_abs_error_s"] == 0.0

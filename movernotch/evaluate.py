"""Recovery metrics. Matching is by proximity, not list position."""
from __future__ import annotations
import numpy as np


def match_errors(pred_idx, true_idx, fs: float = 256.0, tol_s: float = 0.15):
    """Nearest-neighbour match of predictions to references.

    Returns (signed_errors_seconds, n_matched). Each prediction is used at most
    once, so duplicates cannot inflate the detection rate.
    """
    preds = sorted(int(p) for p in pred_idx if p is not None)
    used = set()
    tol = tol_s * fs
    errs = []
    for t in true_idx:
        best, best_d = None, None
        for i, p in enumerate(preds):
            if i in used:
                continue
            d = abs(p - t)
            if best_d is None or d < best_d:
                best, best_d, best_i = p, d, i
        if best is not None and best_d <= tol:
            used.add(best_i)
            errs.append((best - t) / fs)
    return np.array(errs), len(errs)


def summarize(pred_idx, true_idx, fs: float = 256.0,
              tolerances=(0.030, 0.070), tol_s: float = 0.15) -> dict:
    n = len(true_idx)
    errs, found = match_errors(pred_idx, true_idx, fs, tol_s=tol_s)
    n_pred = len([p for p in pred_idx if p is not None])
    out = {
        "n_beats": n,
        "n_predictions": n_pred,
        # Recall alone is gameable at a 150 ms tolerance, so precision too.
        "detection_rate": found / n if n else float("nan"),
        "precision": found / n_pred if n_pred else float("nan"),
        "f1": (2 * found / (n + n_pred)) if (n + n_pred) else float("nan"),
        "mean_abs_error_s": float(np.mean(np.abs(errs))) if len(errs) else float("nan"),
        "median_abs_error_s": float(np.median(np.abs(errs))) if len(errs) else float("nan"),
        "bias_s": float(np.mean(errs)) if len(errs) else float("nan"),
        "scatter_s": float(np.std(errs)) if len(errs) else float("nan"),
    }
    deb = errs - np.mean(errs) if len(errs) else errs
    for tol in tolerances:
        ms = int(tol * 1000)
        out[f"success_within_{ms}ms"] = float(np.mean(np.abs(errs) <= tol)) if len(errs) else float("nan")
        out[f"success_within_{ms}ms_debiased"] = float(np.mean(np.abs(deb) <= tol)) if len(deb) else float("nan")
    return out


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap interval. Small samples are the norm here, so report it."""
    values = np.asarray(values)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))

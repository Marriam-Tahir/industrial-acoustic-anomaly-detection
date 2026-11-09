"""
smart_listener_scoring.py - numpy/scikit-learn only (no torch).
Turns per-window reconstruction errors into recording scores, picks a threshold on the
VALIDATION set, and computes metrics. Shared by the training notebook and the API.

Convention: positive class = abnormal. A recording is called anomalous if score > threshold.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

AGGS = ("mean", "max", "top5mean")


def aggregate(window_err, how):
    """window_err: (n_recordings, n_windows) -> (n_recordings,) recording score."""
    e = np.asarray(window_err, dtype=np.float64)
    if how == "mean":
        return e.mean(axis=1)
    if how == "max":
        return e.max(axis=1)
    if how == "top5mean":
        k = min(5, e.shape[1])
        return np.sort(e, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"unknown aggregation {how!r}; choose from {AGGS}")


def auc(y_true, scores):
    y = np.asarray(y_true)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores))


def _f1_at(normal, abnormal, thr):
    tp = int((abnormal > thr).sum()); fn = len(abnormal) - tp
    fp = int((normal > thr).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def choose_threshold(normal_scores, abnormal_scores, rule="max_f1"):
    """Threshold from validation scores only."""
    n = np.asarray(normal_scores, dtype=np.float64); a = np.asarray(abnormal_scores, dtype=np.float64)
    if len(n) == 0:
        raise ValueError("no normal validation scores")
    if rule == "max_f1":
        if len(a) == 0:
            raise ValueError("rule max_f1 needs abnormal validation scores")
        u = np.unique(np.concatenate([n, a]))
        cands = np.concatenate([[u[0] - 1e-12], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-12]])
        f1s = np.array([_f1_at(n, a, t) for t in cands])
        best = np.flatnonzero(f1s == f1s.max())
        return float(cands[best[len(best) // 2]])      # middle of the best plateau
    if rule in ("p95_normal", "p99_normal"):
        return float(np.percentile(n, 95 if rule == "p95_normal" else 99))
    raise ValueError(f"unknown threshold rule {rule!r}")


def metrics_at(normal_scores, abnormal_scores, thr):
    n = np.asarray(normal_scores, dtype=np.float64); a = np.asarray(abnormal_scores, dtype=np.float64)
    y = np.r_[np.zeros(len(n)), np.ones(len(a))]; s = np.r_[n, a]
    tp = int((a > thr).sum()); fn = len(a) - tp; fp = int((n > thr).sum()); tn = len(n) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    two_classes = len(n) > 0 and len(a) > 0
    return dict(n_normal=len(n), n_abnormal=len(a), threshold=float(thr),
                auc=auc(y, s) if two_classes else float("nan"),
                pauc_fpr_0_1=float(roc_auc_score(y, s, max_fpr=0.1)) if two_classes else float("nan"),
                average_precision=float(average_precision_score(y, s)) if two_classes else float("nan"),
                tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / max(len(y), 1))

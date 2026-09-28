"""Metrics, paired bootstrap, exact McNemar test and Holm correction (Section 3.8).

Every metric is computed from the 4 x 4 confusion matrix, so the 10,000 bootstrap
resamples of the test customers (drawn once and shared by all models) give the
confusion matrix of every resample at once. Point estimates equal scikit-learn's.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from . import config

CLASSES = config.CATEGORIES
OVERALL = ["Acc", "Prec (w)", "Rec (w)", "F1 (w)", "Macro-F1", "Bal. acc"]
CLASS_F1 = [f"F1 {c}" for c in CLASSES]


def metrics_from_confusion(cm) -> dict:
    """cm[..., true, predicted]: one matrix or a stack of bootstrap matrices."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-2, -1))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    w = support / n[..., None]
    out = {"Acc": tp.sum(-1) / n, "Prec (w)": (w * prec).sum(-1), "Rec (w)": (w * rec).sum(-1),
           "F1 (w)": (w * f1).sum(-1), "Macro-F1": f1.mean(-1), "Bal. acc": rec.mean(-1)}
    for j, c in enumerate(CLASSES):
        out[f"F1 {c}"], out[f"P {c}"], out[f"R {c}"] = f1[..., j], prec[..., j], rec[..., j]
    return out


def bootstrap_weights(n: int, n_boot: int = config.N_BOOTSTRAP, rng=None) -> np.ndarray:
    """(n_boot, n) resampling counts: each row resamples the n customers with replacement."""
    rng = rng if rng is not None else np.random.default_rng(config.SEED)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)


class Scored:
    """Point metrics and bootstrap draws of one model on one population (customers in fixed order)."""

    def __init__(self, truth: np.ndarray, pred: np.ndarray, weights: np.ndarray):
        self.y, self.p = truth, pred
        onehot = np.eye(config.NUM_CLASSES ** 2, dtype=np.float32)[truth * config.NUM_CLASSES + pred]
        k = config.NUM_CLASSES
        self.point = {m: float(v) for m, v in metrics_from_confusion(onehot.sum(0).reshape(k, k)).items()}
        self.draws = metrics_from_confusion((weights @ onehot).reshape(-1, k, k))

    def interval(self, metric: str):
        return np.percentile(self.draws[metric], [2.5, 97.5])


def compare(a: Scored, b: Scored, metrics) -> dict:
    """Paired comparison a minus b on the same customers and the same resamples.

    For each metric: difference, 95% percentile interval and the two-sided null-centred
    bootstrap p (share of resamples at least as far from the estimate as the estimate
    is from zero). For accuracy also the exact McNemar test on per-customer correctness.
    """
    out = {}
    for m in metrics:
        d = a.draws[m] - b.draws[m]
        est = a.point[m] - b.point[m]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[f"Diff {m}"], out[f"Diff {m} CI low"], out[f"Diff {m} CI high"] = est, lo, hi
        out[f"Diff {m} p"] = max(float((np.abs(d - est) >= abs(est)).mean()), 1.0 / len(d))
    ca, cb = a.p == a.y, b.p == b.y
    b10, b01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    out["McNemar b"], out["McNemar c"] = b10, b01
    out["McNemar p"] = binomtest(min(b10, b01), b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return out


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adjusted[i] = running
    return adjusted

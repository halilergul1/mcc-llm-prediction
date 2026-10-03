"""Metrics, paired bootstrap, exact McNemar test and Holm correction (Section 3.8).

Point estimates come from scikit-learn. For the 10,000 bootstrap resamples of the test
customers (drawn once and shared by all models), every metric is computed from the confusion
matrix of each resample, which gives the same values as scikit-learn. Predictions are class
codes 0..3; code 4 marks an answer that is not a category, which is always an error.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from . import config

CLASSES = config.CATEGORIES
INVALID = config.NUM_CLASSES      # code of an answer that is not one of the four categories
OVERALL = ["Acc", "Prec (w)", "Rec (w)", "F1 (w)", "Macro-F1", "Bal. acc"]
CLASS_F1 = [f"F1 {c}" for c in CLASSES]
MDE_FACTOR = 2.80                 # z(0.975) + z(0.80) = 1.96 + 0.84


def metrics_from_confusion(cm) -> dict:
    """cm[..., true class, predicted class]: one matrix or a stack of bootstrap matrices.

    An optional last column counts answers that are not a category (errors for every class).
    """
    cm = np.asarray(cm, dtype=float)
    k = cm.shape[-2]
    n = cm.sum(axis=(-2, -1))
    tp = np.diagonal(cm[..., :k], axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm[..., :k].sum(-2)
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


def selection_score(y, p, metric: str = config.SELECTION_METRIC) -> float:
    """Validation score used to pick a configuration: macro-F1, the primary metric of ANALYSIS_PLAN.md."""
    from sklearn.metrics import f1_score

    average = {"Macro-F1": "macro", "F1 (w)": "weighted"}[metric]
    return float(f1_score(y, p, average=average, labels=range(config.NUM_CLASSES), zero_division=0))


def bootstrap_weights(n: int, n_boot: int = config.N_BOOTSTRAP, rng=None) -> np.ndarray:
    """(n_boot, n) resampling counts: each row resamples the n customers with replacement."""
    rng = rng if rng is not None else np.random.default_rng(config.SEED)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)


def point_metrics(y, p) -> dict:
    """All reported metrics of one model, with scikit-learn."""
    labels = list(range(config.NUM_CLASSES))
    prf = lambda average: precision_recall_fscore_support(y, p, labels=labels, average=average, zero_division=0)
    prec_w, rec_w, f1_w, _ = prf("weighted")
    prec, rec, f1, _ = prf(None)
    out = {"Acc": accuracy_score(y, p), "Prec (w)": prec_w, "Rec (w)": rec_w, "F1 (w)": f1_w,
           "Macro-F1": prf("macro")[2], "Bal. acc": rec.mean()}          # mean of the class-wise recalls
    for j, c in enumerate(CLASSES):
        out[f"F1 {c}"], out[f"P {c}"], out[f"R {c}"] = f1[j], prec[j], rec[j]
    return {m: float(v) for m, v in out.items()}


class Scored:
    """Point metrics and bootstrap draws of one model on one population (customers in fixed order)."""

    def __init__(self, truth: np.ndarray, pred: np.ndarray, weights: np.ndarray):
        self.y, self.p = truth, pred
        self.point = point_metrics(truth, pred)
        k = config.NUM_CLASSES
        onehot = np.eye(k * (k + 1), dtype=np.float32)[truth * (k + 1) + pred]
        self.draws = metrics_from_confusion((weights @ onehot).reshape(-1, k, k + 1))

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
        # smallest difference this comparison could detect (two-sided alpha 0.05, power 0.80)
        out[f"Diff {m} MDE"] = MDE_FACTOR * float(np.std(d, ddof=1))
    if a.p is None or b.p is None:                    # averaged models: no per-customer predictions
        out["McNemar b"] = out["McNemar c"] = out["McNemar p"] = np.nan
        return out
    ca, cb = a.p == a.y, b.p == b.y
    b10, b01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    out["McNemar b"], out["McNemar c"] = b10, b01
    out["McNemar p"] = binomtest(min(b10, b01), b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
    return out


def mean_of(scored: list) -> Scored | SimpleNamespace:
    """Average metrics of several models on the same customers, e.g. three training seeds."""
    if len(scored) == 1:
        return scored[0]
    return SimpleNamespace(y=None, p=None,
                           point={m: float(np.mean([s.point[m] for s in scored])) for m in scored[0].point},
                           draws={m: np.mean([s.draws[m] for s in scored], axis=0) for m in scored[0].draws})


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, p[i] * (len(p) - rank)))
        adjusted[i] = running
    return adjusted


# --------------------------------------------------------------------------- probability metrics
def probability_metrics(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> dict:
    """PR-AUC per class (one-vs-rest average precision), their mean, top-2 accuracy, expected calibration
    error of the top class (equal-width bins), and the multiclass Brier score (sum over classes)."""
    from sklearn.metrics import average_precision_score

    y, proba = np.asarray(y), np.asarray(proba, dtype=float)
    out = {}
    for j, c in enumerate(CLASSES):
        out[f"PR-AUC {c}"] = float(average_precision_score(y == j, proba[:, j])) if (y == j).any() else float("nan")
    out["PR-AUC (macro)"] = float(np.nanmean([out[f"PR-AUC {c}"] for c in CLASSES]))
    top2 = np.argsort(-proba, axis=1)[:, :2]
    out["Top-2 acc"] = float((top2 == y[:, None]).any(axis=1).mean())
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    ece = sum((bins == b).mean() * abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean())
              for b in range(n_bins) if (bins == b).any())
    out["ECE"] = float(ece)
    out["Brier"] = float(((proba - np.eye(len(CLASSES))[y]) ** 2).sum(axis=1).mean())
    return out


def reliability_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 15):
    """Rows for a reliability diagram: bin, mean confidence, accuracy, count."""
    import pandas as pd

    conf, pred = np.asarray(proba).max(axis=1), np.asarray(proba).argmax(axis=1)
    bins = np.clip(np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    rows = [{"bin": b, "confidence": float(conf[bins == b].mean()), "accuracy": float((pred[bins == b] == y[bins == b]).mean()),
             "n": int((bins == b).sum())} for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows)


def cost_weighted_threshold(y: np.ndarray, p_class: np.ndarray, cls: int, cost_fn: float, cost_fp: float) -> dict:
    """One-vs-rest operating point for class `cls` that minimises cost_fn * misses + cost_fp * false alarms.
    Choose the threshold on VALIDATION predictions, then apply it unchanged to the test split."""
    y1 = np.asarray(y) == cls
    best = None
    for t in np.unique(np.round(p_class, 4)):
        flag = p_class >= t
        cost = cost_fn * (~flag & y1).sum() + cost_fp * (flag & ~y1).sum()
        if best is None or cost < best["cost"]:
            tp = (flag & y1).sum()
            best = {"threshold": float(t), "cost": float(cost), "recall": float(tp / max(y1.sum(), 1)),
                    "precision": float(tp / max(flag.sum(), 1))}
    return best

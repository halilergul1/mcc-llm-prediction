"""Heuristic baselines (Section 3.6). All return class codes 0..3.

Random, Proportional, Averaging (most frequent category of the window; ties go to the class seen
most recently), last-category and a first-order Markov chain.
"""
import numpy as np

from . import config


def random_predictions(n: int, seed: int = config.SEED) -> np.ndarray:
    """Uniform random prediction over the four classes (one seeded draw)."""
    return np.random.default_rng(seed).choice(config.NUM_CLASSES, size=n)


def proportional_predictions(n: int, shares, seed: int = config.SEED) -> np.ndarray:
    """Predictions sampled from the class shares of the training cohort (one seeded draw)."""
    p = np.asarray(shares, dtype=float)
    return np.random.default_rng(seed).choice(config.NUM_CLASSES, size=n, p=p / p.sum())


def averaging_predictions(codes: np.ndarray, ties: str = "recency") -> np.ndarray:
    """Most frequent category of each input window, (N, X) codes.

    ties="recency": among tied classes, the one seen most recently. ties="lowest": the lowest class code.
    """
    codes = np.asarray(codes)
    counts = np.stack([(codes == k).sum(axis=1) for k in range(config.NUM_CLASSES)], axis=1)
    if ties == "lowest":
        return counts.argmax(axis=1)
    x = codes.shape[1]
    last_seen = np.stack([np.where((codes == k).any(axis=1), x - 1 - np.argmax((codes == k)[:, ::-1], axis=1), -1)
                          for k in range(config.NUM_CLASSES)], axis=1)      # position of the last occurrence
    key = counts * (x + 1) + last_seen          # count first, then recency; positions < x + 1
    return key.argmax(axis=1)


def last_category_predictions(codes: np.ndarray) -> np.ndarray:
    """The category of the most recent input transaction."""
    return np.asarray(codes)[:, -1].copy()


class Markov1:
    """First-order Markov chain: P(next class | last class), estimated on every consecutive pair of the
    training windows (inputs and target), with add-one smoothing."""

    def fit(self, windows: np.ndarray) -> "Markov1":
        w = np.asarray(windows)
        counts = np.ones((config.NUM_CLASSES, config.NUM_CLASSES))
        np.add.at(counts, (w[:, :-1].ravel(), w[:, 1:].ravel()), 1)
        self.transition = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, codes: np.ndarray) -> np.ndarray:
        return self.transition[np.asarray(codes)[:, -1]]

    def predict(self, codes: np.ndarray) -> np.ndarray:
        return self.predict_proba(codes).argmax(axis=1)

"""Random, Proportional and Averaging baselines (Section 3.6). All return class codes 0..3."""
import numpy as np

from . import config


def random_predictions(n: int, seed: int = config.SEED) -> np.ndarray:
    """Uniform random prediction over the four classes (one seeded draw)."""
    return np.random.default_rng(seed).choice(config.NUM_CLASSES, size=n)


def proportional_predictions(n: int, shares, seed: int = config.SEED) -> np.ndarray:
    """Predictions sampled from the class shares of the training cohort (one seeded draw)."""
    p = np.asarray(shares, dtype=float)
    return np.random.default_rng(seed).choice(config.NUM_CLASSES, size=n, p=p / p.sum())


def averaging_predictions(codes: np.ndarray) -> np.ndarray:
    """Most frequent category of each input window, (N, X) codes; ties go to the lowest class code."""
    counts = np.stack([(codes == k).sum(axis=1) for k in range(config.NUM_CLASSES)], axis=1)
    return counts.argmax(axis=1)

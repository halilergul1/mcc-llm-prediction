"""Per-customer prediction files, the input of every metric and test.

One CSV per model and input length, `<name>_last<X>.csv`, or `<name>.csv` for the
length-independent Random and Proportional baselines. Columns:
    customer_id, ground_truth, prediction[, p_<class> for each of the four classes]
A prediction that is not one of the four categories counts as an error. The class probabilities
are written where the model provides them (label scoring of the LLMs, softmax of the baselines).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PATTERN = re.compile(r"^(?P<name>.+?)(?:_last(?P<length>\d+))?\.csv$")


def to_names(codes) -> np.ndarray:
    return np.asarray(config.CATEGORIES)[np.asarray(codes, dtype=int)]


PROB_COLUMNS = [f"p_{c}" for c in config.CATEGORIES]     # optional class probabilities


def write(path: Path, customer_id, ground_truth, prediction, probs=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"customer_id": customer_id, "ground_truth": ground_truth, "prediction": prediction})
    if probs is not None:
        probs = np.asarray(probs, dtype=float)
        for j, col in enumerate(PROB_COLUMNS):
            df[col] = probs[:, j]
    df = df.sort_values("customer_id")
    df.to_csv(path, index=False)
    return path


def file_name(name: str, length: int | None = None) -> str:
    return f"{name}.csv" if length is None else f"{name}_last{length}.csv"


def inventory(folder: Path) -> list[tuple[str, int | None, Path]]:
    """Every prediction file in `folder` as (name, length or None if length-independent, path)."""
    found = []
    for path in sorted(Path(folder).glob("*.csv")):
        m = PATTERN.match(path.name)
        if m:
            length = m.group("length")
            found.append((m.group("name"), int(length) if length else None, path))
    return found


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype={"prediction": str})
    return df.sort_values("customer_id").reset_index(drop=True)

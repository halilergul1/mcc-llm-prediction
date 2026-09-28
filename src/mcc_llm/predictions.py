"""Per-customer prediction files, the input of every metric and test.

One CSV per model and input length, `<name>_last<X>.csv`, or `<name>.csv` for the
length-independent Random and Proportional baselines. Columns:
    customer_id, ground_truth, prediction (raw model output), parsed_prediction (one of the four classes)
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


def write(path: Path, customer_id, ground_truth, parsed, raw=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "customer_id": customer_id,
        "ground_truth": ground_truth,
        "prediction": parsed if raw is None else raw,
        "parsed_prediction": parsed,
    }).sort_values("customer_id")
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
    df = pd.read_csv(path, keep_default_na=False)
    bad = ~df["parsed_prediction"].isin(config.CATEGORIES)
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} predictions outside the four classes")
    return df.sort_values("customer_id").reset_index(drop=True)

"""Input schema, cohort filters, customer draws and the per-customer windows.

Transactions (one CSV file, or a folder of CSV files):
    customer_id, date, amount, mcc[, time]
Demographics (optional, one row per customer):
    customer_id, age, gender, marital_status, education, occupation, income

`mcc` is the four-digit merchant category code. `time` (or another column of
TIE_BREAK_COLUMNS) orders the transactions of one day. `income` is numeric and becomes a
within-bank tercile (low / middle / high), computed over all complete demographic records.
The text fields are used as written, in lower case, in the instruction template.
scripts/export_raw.py writes these files from a bank's raw files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

TRANSACTION_COLUMNS = ["customer_id", "date", "amount", "mcc"]
DEMOGRAPHIC_COLUMNS = ["customer_id", "age", "gender", "marital_status", "education", "occupation", "income"]
TEXT_FIELDS = ["gender", "marital_status", "education", "occupation"]


# --------------------------------------------------------------------------- reading the schema
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files in {path}")
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def require(df: pd.DataFrame, columns: list[str], source) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def map_categories(mcc: pd.Series, mcc_map: dict | None = None) -> pd.Series:
    """Four-digit MCC to one of the four classes; codes not listed are Other."""
    lookup = {int(code): name for name, codes in (mcc_map or config.MCC_CATEGORIES).items() for code in codes}
    return mcc.map(lookup).fillna("Other")


TIE_BREAK_COLUMNS = ("timestamp", "time", "transaction_id", "txn_id")   # the first one present is used


def load_transactions(path, mcc_map: dict | None = None) -> pd.DataFrame:
    """Read transactions, drop incomplete rows, add the class label, sort by customer and date.

    If the file has one of TIE_BREAK_COLUMNS, the transactions of one day are ordered by it; a
    transaction without a value comes first on its day. Transactions that are still tied keep their
    order in the input file. tie_report() counts the customers this can affect.
    """
    df = read_table(path)
    require(df, TRANSACTION_COLUMNS, path)
    extra = [c for c in TIE_BREAK_COLUMNS if c in df.columns][:1]
    df = df[TRANSACTION_COLUMNS + extra].dropna(subset=TRANSACTION_COLUMNS).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["mcc"] = df["mcc"].astype(int)
    df["category"] = map_categories(df["mcc"], mcc_map)
    return df.sort_values(["customer_id", "date"] + extra, kind="stable", na_position="first").reset_index(drop=True)


def tie_report(tx: pd.DataFrame) -> dict:
    """Customers whose most recent date (the target's date) holds two or more transactions."""
    last = tx.groupby("customer_id")["date"].transform("max")
    on_last = tx[tx["date"] == last].groupby("customer_id").size()
    tied = on_last[on_last >= 2]
    mixed = tx[(tx["date"] == last) & tx["customer_id"].isin(tied.index)].groupby("customer_id")["category"].nunique()
    n = tx["customer_id"].nunique()
    return {"customers": n, "tied_last_date": int(len(tied)), "tied_share": len(tied) / n if n else 0.0,
            "tied_with_different_classes": int((mixed > 1).sum()), "tied_ids": np.sort(tied.index.to_numpy())}


def deflate(tx: pd.DataFrame, cpi: dict, base: str) -> pd.DataFrame:
    """Express amounts in the prices of month `base` ("YYYY-MM"); `cpi` maps "YYYY-MM" to an index (e.g. TUIK CPI)."""
    month = tx["date"].dt.strftime("%Y-%m")
    missing = sorted(set(month) - set(cpi))
    if missing:
        raise ValueError(f"CPI missing for months {missing[:5]}...")
    out = tx.copy()
    out["amount"] = out["amount"] * cpi[base] / month.map(cpi).astype(float)
    return out


def load_demographics(path) -> pd.DataFrame:
    """Read the six demographic attributes of complete records, indexed by customer."""
    df = read_table(path)
    require(df, DEMOGRAPHIC_COLUMNS, path)
    df = df[DEMOGRAPHIC_COLUMNS].dropna().drop_duplicates("customer_id").copy()
    df["age"] = df["age"].astype(int)
    for field in TEXT_FIELDS:
        df[field] = df[field].astype(str).str.strip().str.lower()
    df["income_group"] = pd.qcut(df["income"].astype(float), 3, labels=["low", "middle", "high"]).astype(str)
    return df.set_index("customer_id")


# --------------------------------------------------------------------------- cohort and draws
def eligible_customers(tx: pd.DataFrame, demographics: pd.DataFrame | None = None,
                       on_inputs: bool = config.DIVERSITY_ON_INPUTS, min_distinct: int = config.MIN_DISTINCT_CATEGORIES
                       ) -> np.ndarray:
    """Customers that pass the cohort filters of Section 3.2.

    At least 10 transactions, at least `min_distinct` distinct categories, and complete demographics
    when a demographics table is given. With `on_inputs` the categories are counted over the
    9 transactions before the target, so the filter never looks at the target; without it they are
    counted over the last 10, target included (the earlier rule). `min_distinct=1` switches the
    diversity filter off (the unfiltered sensitivity set).
    """
    counts = tx.groupby("customer_id").size()
    active = tx[tx["customer_id"].isin(counts[counts >= config.MIN_TRANSACTIONS].index)]
    recent = active.groupby("customer_id").tail(config.DIVERSITY_WINDOW)          # 9 inputs + target
    if on_inputs:
        recent = recent[recent.groupby("customer_id").cumcount(ascending=False) > 0]   # drop the target
    distinct = recent.groupby("customer_id")["category"].nunique()
    ids = distinct[distinct >= min_distinct].index
    if demographics is not None:
        ids = ids[ids.isin(demographics.index)]
    return np.sort(ids.to_numpy())


def draw(ids, n: int, seed: int = config.SEED) -> np.ndarray:
    """Uniform random draw of `n` customers without replacement."""
    ids = np.sort(np.asarray(ids))
    if n >= len(ids):
        return ids
    return np.sort(np.random.RandomState(seed).choice(ids, size=n, replace=False))


def draw_excluding(ids, n: int | None, exclude=(), seed: int = config.SEED) -> np.ndarray:
    """Uniform draw of `n` customers (all if None) that are not in `exclude`."""
    ids = np.setdiff1d(np.asarray(ids), np.asarray(list(exclude)))
    return ids if n is None else draw(ids, n, seed)


def single_category_share(windows: "Windows") -> dict:
    """Customers whose input window (target excluded) holds a single category."""
    inputs = windows.categories[:, :-1]
    single = (inputs == inputs[:, [0]]).all(axis=1)
    return {"customers": len(windows), "single_category": int(single.sum()),
            "share": float(single.mean()) if len(windows) else 0.0}


def split(ids, val_fraction: float = config.VAL_FRACTION, seed: int = config.SEED):
    """Random split by customer into training and validation sets."""
    ids = np.sort(np.asarray(ids))
    order = np.random.RandomState(seed).permutation(len(ids))
    n_val = int(np.ceil(val_fraction * len(ids)))
    return np.sort(ids[order[n_val:]]), np.sort(ids[order[:n_val]])


def class_shares(tx: pd.DataFrame) -> dict:
    """Share of each class over all transactions."""
    shares = tx["category"].value_counts(normalize=True)
    return {c: float(shares.get(c, 0.0)) for c in config.CATEGORIES}


# --------------------------------------------------------------------------- per-customer windows
@dataclass
class Windows:
    """One sample per customer: the last X + 1 transactions, oldest first.

    Columns 0 .. X-1 are the input and column X, the customer's most recent
    transaction, is the target.
    """

    customer_id: np.ndarray
    categories: np.ndarray   # (N, X + 1) class names
    dates: np.ndarray        # (N, X + 1) datetime64
    amounts: np.ndarray      # (N, X + 1) float

    @property
    def length(self) -> int:
        return self.categories.shape[1] - 1

    @property
    def target(self) -> np.ndarray:
        return self.categories[:, -1]

    def codes(self) -> np.ndarray:
        """Class codes 0..3 of the whole window, (N, X + 1)."""
        lookup = {c: i for i, c in enumerate(config.CATEGORIES)}
        return np.vectorize(lookup.__getitem__)(self.categories).astype(np.int64)

    def __len__(self) -> int:
        return len(self.customer_id)


def last_windows(tx: pd.DataFrame, length: int) -> Windows:
    """Build the evaluation or training samples at input length `length` (the X + 1 rule).

    A customer contributes one sample only if at least `length` + 1 transactions are
    available. There is no padding and no fall-back to a shorter history.
    """
    k = length + 1
    counts = tx.groupby("customer_id")["category"].transform("size")
    w = tx[counts >= k].groupby("customer_id").tail(k)
    block = lambda col: w[col].to_numpy().reshape(-1, k)
    return Windows(
        customer_id=w["customer_id"].to_numpy()[::k],
        categories=block("category"),
        dates=block("date"),
        amounts=block("amount").astype(float),
    )


def variable_windows(tx: pd.DataFrame, lo: int, hi: int, seed: int = config.SEED) -> list[Windows]:
    """One window per customer with a random input length in [lo, min(hi, n - 1)].

    Returns one Windows object per length (the lengths differ, so they cannot share one array).
    """
    counts = tx.groupby("customer_id").size()
    counts = counts[counts >= lo + 1]
    rng = np.random.default_rng(seed)
    chosen = pd.Series([int(rng.integers(lo, min(hi, n - 1) + 1)) for n in counts.to_numpy()], index=counts.index)
    out = []
    for length, ids in chosen.groupby(chosen).groups.items():
        out.append(last_windows(tx[tx["customer_id"].isin(ids)], int(length)))
    return out


# --------------------------------------------------------------------------- prepared splits
def save_split(out_dir: Path, name: str, tx: pd.DataFrame, demographics: pd.DataFrame | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out_dir / f"{name}_transactions.csv", index=False)
    if demographics is not None:
        demographics.loc[np.unique(tx["customer_id"])].to_csv(out_dir / f"{name}_customers.csv")


def load_split(data_dir: str | Path, name: str):
    """Transactions and demographics (or None) of one prepared split."""
    data_dir = Path(data_dir)
    tx = pd.read_csv(data_dir / f"{name}_transactions.csv", parse_dates=["date"])
    customers = data_dir / f"{name}_customers.csv"
    demographics = pd.read_csv(customers, index_col="customer_id") if customers.exists() else None
    return tx, demographics


def load_class_shares(data_dir: str | Path) -> dict:
    return json.loads((Path(data_dir) / "class_shares.json").read_text())

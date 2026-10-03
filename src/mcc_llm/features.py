"""Feature-based baselines: gradient-boosted trees and multinomial logistic regression.

They answer the question the sequential baselines cannot: does a conventional model that sees the SAME
information as the LLM do as well? Two feature sets:

    seq   what Q-seq sees: per-class counts, last and second-to-last class, transactions since each
          class last occurred;
    full  what Q-full sees: seq plus amount statistics (log scale), day gaps and the demographics.

Everything is computed from the input window only (the target is never used).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import Windows

DEMO_CATEGORICAL = ("gender", "marital_status", "education", "occupation", "income_group")


def sequence_features(codes: np.ndarray) -> pd.DataFrame:
    """(N, X) input class codes -> seq features."""
    codes = np.asarray(codes)
    n, x = codes.shape
    cols = {}
    for k, c in enumerate(config.CATEGORIES):
        hit = codes == k
        cols[f"count_{c}"] = hit.sum(axis=1)
        cols[f"share_{c}"] = hit.mean(axis=1)
        last = np.where(hit.any(axis=1), x - 1 - np.argmax(hit[:, ::-1], axis=1), -1)
        cols[f"since_{c}"] = np.where(last >= 0, x - 1 - last, x)        # x = never seen in the window
        cols[f"last_is_{c}"] = (codes[:, -1] == k).astype(int)
        cols[f"second_last_is_{c}"] = (codes[:, -2] == k).astype(int) if x >= 2 else 0
    cols["switches"] = (codes[:, 1:] != codes[:, :-1]).sum(axis=1)
    return pd.DataFrame(cols)


def full_features(windows: Windows, demographics: pd.DataFrame | None) -> pd.DataFrame:
    """seq features plus amounts, dates and demographics of the input window."""
    codes = windows.codes()[:, :-1]
    df = sequence_features(codes)
    amounts = np.log1p(np.maximum(windows.amounts[:, :-1], 0))
    df["amount_mean"], df["amount_median"] = amounts.mean(1), np.median(amounts, 1)
    df["amount_max"], df["amount_last"], df["amount_total"] = amounts.max(1), amounts[:, -1], np.log1p(
        np.maximum(windows.amounts[:, :-1], 0).sum(1))
    days = windows.dates[:, :-1].astype("datetime64[D]").astype(np.int64)
    gaps = np.diff(days, axis=1)
    df["gap_mean"], df["gap_last"], df["span_days"] = gaps.mean(1), gaps[:, -1], days[:, -1] - days[:, 0]
    df["weekday_last"] = (days[:, -1] + 3) % 7                              # 1970-01-01 was a Thursday
    if demographics is not None:
        demo = demographics.loc[windows.customer_id]
        df["age"] = demo["age"].to_numpy()
        for f in DEMO_CATEGORICAL:
            df[f] = pd.Categorical(demo[f].astype(str).to_numpy())
    return df


def dummy_columns(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Categorical columns of the training table and the columns of its dummy encoding."""
    cats = [c for c in train.columns if isinstance(train[c].dtype, pd.CategoricalDtype)]
    return cats, list(pd.get_dummies(train, columns=cats, dtype=float).columns)


def one_hot(df: pd.DataFrame, cats: list[str], columns: list[str]) -> pd.DataFrame:
    """Dummy-encode `df` with the training levels (for logistic regression); unseen levels become zeros."""
    return pd.get_dummies(df, columns=cats, dtype=float).reindex(columns=columns, fill_value=0.0)


def fit_lightgbm(X_tr, y_tr, X_va, y_va, params: dict, seed: int = config.SEED):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(objective="multiclass", num_class=config.NUM_CLASSES, n_estimators=2000,
                               random_state=seed, verbose=-1, **params)
    stop = [lgb.early_stopping(50, verbose=False)]
    try:
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=stop)
    except TypeError:                                 # LightGBM before 4.7 takes the validation data as eval_set
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=stop)
    return model


def fit_logistic(X_tr, y_tr, C: float, class_weight=None, seed: int = config.SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight=class_weight,
                                                               random_state=seed))
    return model.fit(X_tr, y_tr)


LGBM_GRID = [{"learning_rate": lr, "num_leaves": nl, "min_child_samples": mc, "class_weight": cw}
             for lr in (0.05, 0.1) for nl in (15, 31, 63) for mc in (20, 100) for cw in (None, "balanced")]
LOGIT_GRID = [{"C": C, "class_weight": cw} for C in (0.01, 0.1, 1.0, 10.0) for cw in (None, "balanced")]

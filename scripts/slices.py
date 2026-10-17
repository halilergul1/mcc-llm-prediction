"""Scores by slice of the customer population, in-bank against cross-bank (exploratory; Appendix F).

Each test customer at length 9 is placed in a slice by (i) the span in days of its last ten transactions
and (ii) the number of distinct categories among its nine input transactions. For each slice the script
gives the number of customers and the scores of a few models on both test splits, so one can see whether
the cross-bank scores follow the composition of Bank B rather than the bank itself. Descriptive: no tests.
Seed means where a model has several seeds.

    python scripts/slices.py --predictions results/predictions --data data/processed --out results/local/slices
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, data, evaluation, predictions

SEED = re.compile(r"^(?P<base>.+?)-seed(?P<seed>\d+)$")
CODE = {c: i for i, c in enumerate(config.CATEGORIES)}
SPAN_EDGES = [-1, 7, 30, 90, np.inf]
SPAN_LABELS = ["0-7 days", "8-30 days", "31-90 days", "over 90 days"]


def runs_of(folder: Path, base: str, length: int) -> list[pd.DataFrame]:
    out = []
    for name, x, path in predictions.inventory(folder):
        m = SEED.match(name)
        if x == length and (name == base or (m and m.group("base") == base)):
            out.append(predictions.read(path).set_index("customer_id"))
    return out


def slice_table(tx: pd.DataFrame) -> pd.DataFrame:
    w = data.last_windows(tx, config.TRAIN_LENGTH)
    codes = w.codes()[:, :-1]
    span = (w.dates[:, -1] - w.dates[:, 0]) / np.timedelta64(1, "D")
    distinct = np.stack([(codes == k).any(1) for k in range(config.NUM_CLASSES)], 1).sum(1)
    return pd.DataFrame({"span": pd.cut(span.astype(float), SPAN_EDGES, labels=SPAN_LABELS).astype(str),
                         "distinct input categories": distinct.astype(str)}, index=w.customer_id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--splits", nargs="+", default=["bank_a_test", "bank_b_confirm"])
    p.add_argument("--models", nargs="+", default=["qwen-full", "averaging", "gbdt-full", "sasrec"])
    p.add_argument("--out", type=Path, default=Path("results/local/slices"))
    args = p.parse_args()
    rows = []
    for split in args.splits:
        tx, _ = data.load_split(args.data, split)
        sl = slice_table(tx)
        for variable in sl.columns:
            for level, ids in sl.groupby(variable).groups.items():
                for model in args.models:
                    runs = runs_of(args.predictions / split, model, config.TRAIN_LENGTH)
                    if not runs:
                        continue
                    scores = []
                    for df in runs:
                        d = df.loc[ids]
                        y = d["ground_truth"].map(CODE).to_numpy()
                        pr = d["prediction"].map(CODE).fillna(evaluation.INVALID).astype(int).to_numpy()
                        scores.append(evaluation.point_metrics(y, pr))
                    mean = pd.DataFrame(scores).mean()
                    rows.append({"split": split, "variable": variable, "level": level, "N": len(ids), "model": model,
                                 "seeds": len(runs), "Acc": mean["Acc"], "F1 (w)": mean["F1 (w)"],
                                 "Macro-F1": mean["Macro-F1"], "F1 Clothing": mean["F1 Clothing"]})
            share = sl[variable].value_counts(normalize=True)
            for level, s in share.items():
                rows.append({"split": split, "variable": variable, "level": level, "N": int((sl[variable] == level).sum()),
                             "model": "share of customers (%)", "seeds": np.nan, "Acc": 100 * s})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "slices.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

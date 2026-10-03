"""Integrity checks of the data and of the prediction files. Read-only: nothing is overwritten.

    # single-category input windows and ties on the target date, per split
    python scripts/p0_checks.py cohort --data data/processed
    # the merchant category codes behind each class, per bank
    python scripts/p0_checks.py mcc --transactions raw/bank_a/transactions.csv raw/bank_b/transactions.csv
    # customers shared by two ID lists (for example two test draws), and the age of every prediction file
    python scripts/p0_checks.py overlap --a old/first_test_customers.csv --b old/earlier_test_customers.csv \
        --population 11556
    # answers outside the four labels, per model and length
    python scripts/p0_checks.py invalid --predictions results/predictions/bank_b_dev

An ID file is any CSV with a customer_id column (a *_transactions.csv or *_customers.csv split file works).
"""
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, data, predictions


def ids_of(path: Path) -> np.ndarray:
    return np.unique(pd.read_csv(path, usecols=["customer_id"])["customer_id"].to_numpy())


def overlap(args):
    a, b = ids_of(args.a), ids_of(args.b)
    common = np.intersect1d(a, b)
    print(f"list a: {len(a):,} customers; list b: {len(b):,}; in both: {len(common):,}")
    if args.population:
        expected = len(a) * len(b) / args.population
        print(f"expected overlap of two independent uniform draws from {args.population:,}: about {expected:.0f}")
        print("   far above that -> the two draws are not independent (same seed?); far below -> check the filters")
    if args.predictions:
        rows = [{"file": path.name, "modified": dt.datetime.fromtimestamp(path.stat().st_mtime)}
                for name, length, path in predictions.inventory(args.predictions)]
        print("\nprediction files, oldest first:")
        print(pd.DataFrame(rows).sort_values("modified").to_string(index=False))


def invalid(args):
    rows = []
    for name, length, path in predictions.inventory(args.predictions):
        df = pd.read_csv(path, keep_default_na=False, dtype=str)
        bad = ~df["prediction"].isin(config.CATEGORIES)
        rows.append({"model": name, "length": length, "N": len(df), "not a category": int(bad.sum()),
                     "share (%)": round(100 * float(bad.mean()), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


def cohort(args):
    for split in args.splits:
        try:
            tx, _ = data.load_split(args.data, split)
        except FileNotFoundError:
            continue
        for x in config.EVAL_LENGTHS:
            w = data.last_windows(tx, x)
            s = data.single_category_share(w)
            print(f"{split:18s} length {x:2d}: {s['customers']:6,} customers, single-category input "
                  f"{s['single_category']:5,} ({100 * s['share']:.1f}%)")
        t = data.tie_report(tx)
        print(f"{split:18s} ties on the target date: {t['tied_last_date']:,} of {t['customers']:,} "
              f"({100 * t['tied_share']:.1f}%); with two or more classes on that date: {t['tied_with_different_classes']:,}")


def mcc(args):
    """Class shares and the most frequent codes of each class, per file (for the table of codes per class)."""
    for path in args.transactions:
        tx = data.read_table(path)
        codes = tx["mcc"].dropna().astype(int)
        classes = data.map_categories(codes)
        print(f"\n{path}: {len(codes):,} transactions")
        for name in config.CATEGORIES:
            inside = codes[classes == name]
            top = (inside.value_counts(normalize=True).head(args.top) * 100).round(1)
            print(f"  {name}: {100 * len(inside) / len(codes):.1f}% of transactions, {inside.nunique()} codes; "
                  f"largest: " + ", ".join(f"{code} {share}%" for code, share in top.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlap"); o.add_argument("--a", type=Path, required=True); o.add_argument("--b", type=Path, required=True)
    o.add_argument("--population", type=int); o.add_argument("--predictions", type=Path)
    i = sub.add_parser("invalid"); i.add_argument("--predictions", type=Path, required=True)
    c = sub.add_parser("cohort"); c.add_argument("--data", type=Path, default=Path("data/processed"))
    c.add_argument("--splits", nargs="+", default=["train", "val", "bank_a_test", "bank_b_dev", "bank_b_confirm",
                                                   "bank_b_unfiltered"])
    m = sub.add_parser("mcc"); m.add_argument("--transactions", type=Path, nargs="+", required=True)
    m.add_argument("--top", type=int, default=6, help="codes listed per class")
    args = p.parse_args()
    {"overlap": overlap, "invalid": invalid, "cohort": cohort, "mcc": mcc}[args.cmd](args)


if __name__ == "__main__":
    main()

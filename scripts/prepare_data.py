"""Build the cohorts and the evaluation splits (Sections 3.1 and 3.2).

Bank A (training bank): eligible customers -> random draw of 50,000 -> 80/20 train/val split;
                        a further 10,000 never-drawn eligible customers -> bank_a_test (in-bank reference).
Bank B (test bank):     bank_b_dev      = the customers of --dev-ids that are eligible (smoke tests only);
                        bank_b_confirm  = every other eligible customer, minus --exclude (primary evaluation);
                        bank_b_unfiltered = customers with >= 10 transactions and NO diversity filter, minus
                                            --exclude and minus the dev customers (sensitivity analysis).
Eligible: complete demographics, at least 10 transactions, at least 2 distinct categories among the
9 transactions BEFORE the target (config.DIVERSITY_ON_INPUTS). --earlier-rule counts them over
the last 10 transactions, target included, and draws a single test split (to rebuild earlier cohorts).

Example:
    python scripts/prepare_data.py \
        --train-transactions raw/bank_a/transactions.csv --train-demographics raw/bank_a/demographics.csv \
        --test-transactions raw/bank_b/transactions.csv --test-demographics raw/bank_b/demographics.csv \
        --dev-ids old/earlier_test_customers.csv --exclude old/earlier_test_customers.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, data


def cohort(transactions, demographics, mcc_map, bank, on_inputs, min_distinct=config.MIN_DISTINCT_CATEGORIES):
    tx = data.load_transactions(transactions, mcc_map)
    demo = data.load_demographics(demographics) if demographics else None
    ids = data.eligible_customers(tx, demo, on_inputs=on_inputs, min_distinct=min_distinct)
    print(f"{bank}: {tx['customer_id'].nunique():,} customers in the input, {len(ids):,} eligible")
    return tx, demo, ids


def read_ids(paths):
    ids = [pd.read_csv(p, usecols=["customer_id"])["customer_id"].to_numpy() for p in paths or []]
    return np.unique(np.concatenate(ids)) if ids else np.array([], dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-transactions", required=True, help="training bank transactions (file or folder of CSVs)")
    p.add_argument("--train-demographics", help="training bank demographics (optional)")
    p.add_argument("--test-transactions", required=True, help="test bank transactions (file or folder of CSVs)")
    p.add_argument("--test-demographics", help="test bank demographics (optional)")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--n-train-pool", type=int, default=config.N_TRAIN_POOL, help="customers drawn from the training bank")
    p.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    p.add_argument("--n-bank-a-test", type=int, default=config.N_BANK_A_TEST)
    p.add_argument("--dev-ids", type=Path, help="CSV with customer_id: test customers of an earlier evaluation")
    p.add_argument("--exclude", type=Path, nargs="*", default=[],
                   help="CSVs with customer_id to keep out of the confirmation set (every earlier test set)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--mcc-map", type=Path, help="JSON {class: [MCC, ...]} replacing the default map in config.py")
    p.add_argument("--cpi", type=Path, help="CSV with columns month (YYYY-MM) and index: deflate the amounts (default: nominal)")
    p.add_argument("--deflate-to", default=config.DEFLATE_TO, help="base month YYYY-MM for --cpi (e.g. 2013-04)")
    p.add_argument("--earlier-rule", action="store_true",
                   help="categories counted over the last 10 transactions, target included; one test split")
    p.add_argument("--n-test", type=int, default=config.N_TEST, help="--earlier-rule only: size of the test draw")
    args = p.parse_args()
    mcc_map = json.loads(args.mcc_map.read_text()) if args.mcc_map else None
    on_inputs = not args.earlier_rule

    tx_a, demo_a, ids_a = cohort(args.train_transactions, args.train_demographics, mcc_map, "training bank", on_inputs)
    cpi = None
    if args.cpi:
        if not args.deflate_to:
            raise SystemExit("--cpi needs --deflate-to YYYY-MM")
        table = pd.read_csv(args.cpi, dtype={"month": str})
        cpi = dict(zip(table["month"], table["index"].astype(float)))
        tx_a = data.deflate(tx_a, cpi, args.deflate_to)
        print(f"amounts deflated to {args.deflate_to} prices with {args.cpi}")
    pool = data.draw(ids_a, args.n_train_pool, args.seed)
    train_ids, val_ids = data.split(pool, args.val_fraction, args.seed)
    tx_b, demo_b, ids_b = cohort(args.test_transactions, args.test_demographics, mcc_map, "test bank", on_inputs)
    if cpi is not None:
        tx_b = data.deflate(tx_b, cpi, args.deflate_to)
    splits = {"train": (tx_a, demo_a, train_ids), "val": (tx_a, demo_a, val_ids)}
    if args.earlier_rule:
        splits["test"] = (tx_b, demo_b, data.draw(ids_b, args.n_test, args.seed))
    else:
        splits["bank_a_test"] = (tx_a, demo_a, data.draw_excluding(ids_a, args.n_bank_a_test, pool, args.seed))
        dev = np.intersect1d(ids_b, read_ids([args.dev_ids] if args.dev_ids else []))
        excluded = np.union1d(read_ids(args.exclude), dev)
        splits["bank_b_dev"] = (tx_b, demo_b, dev)
        splits["bank_b_confirm"] = (tx_b, demo_b, data.draw_excluding(ids_b, None, excluded))
        _, _, loose = cohort(args.test_transactions, args.test_demographics, mcc_map, "test bank, no diversity filter",
                             on_inputs, min_distinct=1)
        splits["bank_b_unfiltered"] = (tx_b, demo_b, data.draw_excluding(loose, None, excluded))

    for name, (tx, demo, ids) in splits.items():
        data.save_split(args.out, name, tx[tx["customer_id"].isin(ids)], demo)
        print(f"{name:18s} {len(ids):7,} customers")
    shares = data.class_shares(tx_a[tx_a["customer_id"].isin(ids_a)])      # class frequencies of the training cohort
    (args.out / "class_shares.json").write_text(json.dumps(shares, indent=1))
    print("training cohort class shares:", {c: round(100 * s, 1) for c, s in shares.items()})
    summary = {
        "seed": args.seed, "diversity_on_inputs": on_inputs, "min_transactions": config.MIN_TRANSACTIONS,
        "min_distinct_categories": config.MIN_DISTINCT_CATEGORIES, "categories": list(config.CATEGORIES),
        "amounts": f"deflated to {args.deflate_to}" if cpi is not None else "nominal",
        "eligible": {"training bank": len(ids_a), "test bank": len(ids_b)},
        "customers": {name: len(ids) for name, (_, _, ids) in splits.items()},
        "excluded_from_confirmation": {Path(f).name: len(read_ids([f])) for f in args.exclude},
    }
    (args.out / "splits.json").write_text(json.dumps(summary, indent=1))
    assert not set(train_ids) & set(val_ids)
    if not args.earlier_rule:
        assert not set(splits["bank_a_test"][2]) & set(pool), "in-bank test overlaps training or validation"
        assert not set(splits["bank_b_confirm"][2]) & set(excluded), "confirmation set overlaps an earlier test set"
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()

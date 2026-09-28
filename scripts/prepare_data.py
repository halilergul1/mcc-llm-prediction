"""Build the training, validation and test cohorts (Sections 3.1 and 3.2).

Training bank (Bank A): eligible customers -> random draw of 50,000 -> 80/20 split by customer.
Test bank (Bank B):     eligible customers -> random draw of 1,000.
Eligible: complete demographics, at least 10 transactions, and at least 2 distinct
categories among the last 10 transactions.

Example:
    python scripts/prepare_data.py \
        --train-transactions raw/bank_a/transactions.csv --train-demographics raw/bank_a/demographics.csv \
        --test-transactions raw/bank_b/transactions.csv --test-demographics raw/bank_b/demographics.csv
"""
import argparse
import json
from pathlib import Path

from mcc_llm import config, data


def cohort(transactions, demographics, mcc_map, bank):
    tx = data.load_transactions(transactions, mcc_map)
    demo = data.load_demographics(demographics) if demographics else None
    ids = data.eligible_customers(tx, demo)
    print(f"{bank}: {tx['customer_id'].nunique():,} customers in the input, {len(ids):,} eligible")
    return tx[tx["customer_id"].isin(ids)], demo, ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-transactions", required=True, help="training bank transactions (file or folder of CSVs)")
    p.add_argument("--train-demographics", help="training bank demographics (optional)")
    p.add_argument("--test-transactions", required=True, help="test bank transactions (file or folder of CSVs)")
    p.add_argument("--test-demographics", help="test bank demographics (optional)")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--n-train-pool", type=int, default=config.N_TRAIN_POOL, help="customers drawn from the training bank")
    p.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    p.add_argument("--n-test", type=int, default=config.N_TEST, help="customers drawn from the test bank")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--mcc-map", type=Path, help="JSON {class: [MCC, ...]} replacing the default map in config.py")
    args = p.parse_args()
    mcc_map = json.loads(args.mcc_map.read_text()) if args.mcc_map else None

    tx_a, demo_a, ids_a = cohort(args.train_transactions, args.train_demographics, mcc_map, "training bank")
    shares = data.class_shares(tx_a)                       # class frequencies of the training cohort
    train_ids, val_ids = data.split(data.draw(ids_a, args.n_train_pool, args.seed), args.val_fraction, args.seed)
    tx_b, demo_b, ids_b = cohort(args.test_transactions, args.test_demographics, mcc_map, "test bank")
    test_ids = data.draw(ids_b, args.n_test, args.seed)

    for name, tx, demo, ids in [("train", tx_a, demo_a, train_ids), ("val", tx_a, demo_a, val_ids),
                                ("test", tx_b, demo_b, test_ids)]:
        data.save_split(args.out, name, tx[tx["customer_id"].isin(ids)], demo)
    (args.out / "class_shares.json").write_text(json.dumps(shares, indent=1))

    print(f"training / validation / test customers: {len(train_ids):,} / {len(val_ids):,} / {len(test_ids):,}")
    print("training cohort class shares:", {c: round(100 * s, 1) for c, s in shares.items()})
    test_tx, _ = data.load_split(args.out, "test")
    print("test customers per input length:", {x: len(data.last_windows(test_tx, x)) for x in config.EVAL_LENGTHS})
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()

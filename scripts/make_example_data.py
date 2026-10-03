"""Write small synthetic data in the input schema, to try the pipeline without bank data.

All values are random. The numbers they produce say nothing about the paper's results.

    python scripts/make_example_data.py --out data/example
    python scripts/prepare_data.py --train-transactions data/example/bank_a/transactions.csv \
        --train-demographics data/example/bank_a/demographics.csv \
        --test-transactions data/example/bank_b/transactions.csv \
        --test-demographics data/example/bank_b/demographics.csv --n-train-pool 2000 --n-bank-a-test 500 \
        --dev-ids data/example/bank_b/earlier_test_customers.csv --exclude data/example/bank_b/earlier_test_customers.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MCC = {"Food and grocery": [5411, 5462, 5499], "Clothing": [5651, 5691], "Gas stations": [5541, 5542],
       "Other": [4812, 5732, 5812, 5912]}
PROFILE = {
    "gender": ["male", "female"],
    "marital_status": ["married", "single", "divorced"],
    "education": ["primary school", "high school", "university"],
    "occupation": ["private sector employee", "public sector employee", "self-employed", "retired"],
}


def bank(n_customers, start, days, first_id, rng):
    classes = list(MCC)
    tx, demo = [], []
    for cid in range(first_id, first_id + n_customers):
        taste = rng.dirichlet(np.full(len(classes), 0.6))          # each customer has a favourite mix
        n = rng.integers(8, 40)
        dates = start + pd.to_timedelta(np.sort(rng.integers(0, days, n)), unit="D")
        current = rng.choice(len(classes), p=taste)
        for d in dates:
            if rng.random() > 0.55:                                  # habits persist from one purchase to the next
                current = rng.choice(len(classes), p=taste)
            cls = classes[current]
            clock = f"{rng.integers(7, 23):02d}:{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}"
            tx.append((cid, d.date().isoformat(), clock, round(float(rng.lognormal(3.3, 1.0)), 2), int(rng.choice(MCC[cls]))))
        demo.append((cid, int(rng.integers(19, 75)), *(str(rng.choice(v)) for v in PROFILE.values()),
                     round(float(rng.lognormal(8.5, 0.5)), 2)))
    tx = pd.DataFrame(tx, columns=["customer_id", "date", "time", "amount", "mcc"])
    demo = pd.DataFrame(demo, columns=["customer_id", "age", *PROFILE, "income"])
    return tx, demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/example"))
    p.add_argument("--n-bank-a", type=int, default=3000)
    p.add_argument("--n-bank-b", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    for name, n, start, days, first_id in [("bank_a", args.n_bank_a, "2013-04-01", 91, 1_000_000),
                                           ("bank_b", args.n_bank_b, "2014-07-01", 365, 9_000_000)]:
        tx, demo = bank(n, pd.Timestamp(start), days, first_id, rng)
        (args.out / name).mkdir(parents=True, exist_ok=True)
        tx.to_csv(args.out / name / "transactions.csv", index=False)
        demo.to_csv(args.out / name / "demographics.csv", index=False)
        print(f"{name}: {len(demo):,} customers, {len(tx):,} transactions -> {args.out / name}/")
    # stands for the test customers of an earlier evaluation: they become bank_b_dev and stay out of bank_b_confirm
    earlier = demo["customer_id"].sample(min(100, len(demo)), random_state=args.seed).sort_values()
    earlier.to_frame().to_csv(args.out / "bank_b" / "earlier_test_customers.csv", index=False)


if __name__ == "__main__":
    main()

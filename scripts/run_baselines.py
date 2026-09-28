"""Random, Proportional and Averaging baselines on the test customers (Section 3.6).

Random and Proportional do not depend on the input length and are written once;
Averaging, the most frequent category of the input window, is written per length.
"""
import argparse
from pathlib import Path

from mcc_llm import baselines, config, data, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/predictions"))
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    test_tx, _ = data.load_split(args.data, "test")
    shares = data.load_class_shares(args.data)
    everyone = data.last_windows(test_tx, 1)        # final transaction of every test customer = target
    n = len(everyone)
    draws = {"random": baselines.random_predictions(n, args.seed),
             "proportional": baselines.proportional_predictions(n, [shares[c] for c in config.CATEGORIES], args.seed)}
    for name, codes in draws.items():
        path = predictions.write(args.out / predictions.file_name(name), everyone.customer_id, everyone.target,
                                 predictions.to_names(codes))
        print(f"{name}: {path}")
    for x in args.lengths:
        w = data.last_windows(test_tx, x)
        pred = baselines.averaging_predictions(w.codes()[:, :-1])
        path = predictions.write(args.out / predictions.file_name("averaging", x), w.customer_id, w.target,
                                 predictions.to_names(pred))
        print(f"averaging, length {x} (N = {len(w):,}): {path}")


if __name__ == "__main__":
    main()

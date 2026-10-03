"""Heuristic baselines on the evaluation splits (Section 3.6).

Random and Proportional do not depend on the input length and are written once per split;
Averaging (most frequent category; ties go to the class seen most recently), last-category and the
first-order Markov chain (fitted on the training windows) are written per length.

    python scripts/run_baselines.py
After the freeze, the confirmation splits:
    python scripts/run_baselines.py --splits bank_b_confirm bank_b_unfiltered
"""
import argparse
from pathlib import Path

from mcc_llm import baselines, config, data, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/predictions"))
    p.add_argument("--splits", nargs="+", default=list(config.PRE_SPLITS),
                   help="evaluation splits to score (default: the splits that may be scored before the freeze)")
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--averaging-ties", choices=["recency", "lowest"], default="recency")
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    shares = data.load_class_shares(args.data)
    train_tx, _ = data.load_split(args.data, "train")
    markov = baselines.Markov1().fit(data.last_windows(train_tx, config.TRAIN_LENGTH).codes())
    for split in args.splits:
        tx, _ = data.load_split(args.data, split)
        folder = args.out if args.splits == ["test"] else args.out / split
        everyone = data.last_windows(tx, 1)        # final transaction of every customer = target
        n = len(everyone)
        draws = {"random": baselines.random_predictions(n, args.seed),
                 "proportional": baselines.proportional_predictions(n, [shares[c] for c in config.CATEGORIES], args.seed)}
        for name, codes in draws.items():
            predictions.write(folder / predictions.file_name(name), everyone.customer_id, everyone.target,
                              predictions.to_names(codes))
        for x in args.lengths:
            w = data.last_windows(tx, x)
            if len(w) == 0:
                continue
            inputs = w.codes()[:, :-1]
            rules = {"averaging": baselines.averaging_predictions(inputs, args.averaging_ties),
                     "last-category": baselines.last_category_predictions(inputs)}
            for name, pred in rules.items():
                predictions.write(folder / predictions.file_name(name, x), w.customer_id, w.target,
                                  predictions.to_names(pred))
            proba = markov.predict_proba(inputs)
            predictions.write(folder / predictions.file_name("markov1", x), w.customer_id, w.target,
                              predictions.to_names(proba.argmax(1)), probs=proba)
            print(f"{split}, length {x} (N = {len(w):,}): averaging, last-category, markov1 written")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()

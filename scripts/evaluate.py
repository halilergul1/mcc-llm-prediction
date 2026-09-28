"""Metrics and paired significance tests from the per-customer prediction files (Section 3.8).

metrics.csv       every model at every length: accuracy, weighted precision, recall and F1,
                  macro-F1, balanced accuracy and class-wise F1, with 95% bootstrap intervals
significance.csv  every LLM against every baseline at every length: paired difference, 95%
                  interval and null-centred bootstrap p for each F1 metric and balanced accuracy,
                  exact McNemar p for accuracy, and Holm-adjusted p within one LLM, one length
                  and one metric over the baselines

A model is scored at length X on the customers with at least X + 1 transactions, so the
population can change with X; --fixed-population scores every length on the customers of
the longest length. 10,000 bootstrap resamples of the customers are drawn once per
population and shared by every model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, evaluation, predictions

TESTED = ["F1 (w)", "Macro-F1", "Bal. acc"] + evaluation.CLASS_F1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--out", type=Path, default=Path("results/metrics"))
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--llms", nargs="+", default=["mistral-full", "qwen-seq", "qwen-full"])
    p.add_argument("--baselines", nargs="+", default=["averaging", "lstm", "cnn", "sasrec", "dros-sasrec"])
    p.add_argument("--fixed-population", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    files = predictions.inventory(args.predictions)
    if not files:
        raise SystemExit(f"no prediction files in {args.predictions}")
    tables = {(name, length): predictions.read(path) for name, length, path in files}
    per_length = {x: sorted(n for n, length in tables if length == x) for x in args.lengths}
    fixed = population(tables, max(args.lengths)) if args.fixed_population else None
    rng, weights = np.random.default_rng(args.seed), {}

    def score(name, x):
        df = tables.get((name, x), tables.get((name, None)))
        customers = fixed if fixed is not None else population(tables, x)
        df = df.set_index("customer_id").loc[customers]
        key = tuple(customers)
        if key not in weights:
            weights[key] = evaluation.bootstrap_weights(len(customers), args.n_bootstrap, rng)
        code = {c: i for i, c in enumerate(config.CATEGORIES)}
        return evaluation.Scored(df["ground_truth"].map(code).to_numpy(), df["parsed_prediction"].map(code).to_numpy(),
                                 weights[key])

    independent = sorted(n for n, length in tables if length is None)
    rows, tests = [], []
    for x in args.lengths:
        scored = {name: score(name, x) for name in independent + per_length[x]}
        for name, s in scored.items():
            row = {"model": name, "length": x, "N": len(s.y)}
            for m in evaluation.OVERALL + evaluation.CLASS_F1:
                lo, hi = s.interval(m)
                row.update({m: s.point[m], f"{m} CI low": lo, f"{m} CI high": hi})
            rows.append(row)
        for llm_name in [n for n in args.llms if n in scored]:
            family = [{"length": x, "N": len(scored[llm_name].y), "llm": llm_name, "baseline": b,
                       **evaluation.compare(scored[llm_name], scored[b], TESTED)}
                      for b in args.baselines if b in scored]
            if not family:
                continue
            family = pd.DataFrame(family)
            family["McNemar p (Holm)"] = evaluation.holm(family["McNemar p"])
            for m in TESTED:
                family[f"Diff {m} p (Holm)"] = evaluation.holm(family[f"Diff {m} p"])
            tests.append(family)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    print(metrics[["model", "length", "N", "Acc", "F1 (w)", "Macro-F1", "F1 Clothing"]].round(3).to_string(index=False))
    if tests:
        significance = pd.concat(tests, ignore_index=True)
        significance.to_csv(args.out / "significance.csv", index=False)
        view = significance[["length", "llm", "baseline", "Diff F1 (w)", "Diff F1 (w) p (Holm)", "McNemar p (Holm)"]]
        print("\n" + view.round(4).to_string(index=False))
    print(f"\nwritten to {args.out}/")


def population(tables, length):
    """Customers of the length-specific prediction files at `length` (identical across models)."""
    sets = [tuple(df["customer_id"]) for (name, x), df in tables.items() if x == length]
    if not sets:
        raise SystemExit(f"no prediction files for length {length}")
    if len(set(sets)) > 1:
        raise SystemExit(f"prediction files at length {length} cover different customers")
    return list(sets[0])


if __name__ == "__main__":
    main()

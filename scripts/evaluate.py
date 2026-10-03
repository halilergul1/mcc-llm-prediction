"""Metrics and paired tests from the per-customer prediction files (Section 3.8).

Outputs in --out:
    metrics.csv       every model at every length: accuracy, weighted precision, recall and F1,
                      macro-F1, balanced accuracy, class-wise precision, recall and F1, 95% intervals
    confusion.csv     confusion matrix of every model at every length (rows: true class; the last
                      column counts answers that are not a category)
    significance.csv  every LLM against every baseline at every length: difference, 95% interval and
                      null-centred bootstrap p for the F1 metrics and balanced accuracy, exact McNemar p
                      for accuracy, and Holm-adjusted p over the baselines (one LLM, length and metric)
    pairs.csv         --pairs: single paired comparisons (unadjusted p), e.g. fine-tuned against raw,
                      Q-full against Q-seq, Q-seq against Q-neutral, leave-one-field-out, length effects
    factorial.csv     main effects and interactions of the 2^3 context factorial at length 9, written
                      when the eight condition files are present

A model is scored at length X on the customers with at least X + 1 transactions, so the population
changes with X; --fixed-population scores every length on the customers of the longest length. A
comparison across two lengths uses the customers present at both. The 10,000 bootstrap resamples are
drawn once per population and shared by every model and metric.

Pair syntax, A:B = A minus B. A side is a model name, optionally with @length; several names joined
by commas are averaged (for example three training seeds):
    qwen-full:qwen-raw   qwen-full@9:qwen-full@14   qwen-full,qwen-full-seed2:qwen-seq,qwen-seq-seed2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, evaluation, predictions

TESTED = ["F1 (w)", "Macro-F1", "Bal. acc"] + evaluation.CLASS_F1
REPORTED = evaluation.OVERALL + [f"{k} {c}" for c in config.CATEGORIES for k in ("P", "R", "F1")]
PAIR_METRICS = ["Acc", "F1 (w)", "Macro-F1"] + evaluation.CLASS_F1
FACTORS = ("amounts", "dates", "demographics")
FACTORIAL = {(0, 0, 0): "qwen-seq", (1, 0, 0): "qwen-seq+amounts", (0, 1, 0): "qwen-seq+dates",
             (0, 0, 1): "qwen-seq+demographics", (1, 1, 0): "qwen-seq+amounts+dates",
             (1, 0, 1): "qwen-seq+amounts+demographics", (0, 1, 1): "qwen-seq+dates+demographics",
             (1, 1, 1): "qwen-full"}
CODE = {c: i for i, c in enumerate(config.CATEGORIES)}
NOT_A_CATEGORY = "not a category"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--out", type=Path, default=Path("results/metrics"))
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--llms", nargs="+", default=["mistral-full", "qwen-seq", "qwen-full"])
    p.add_argument("--baselines", nargs="+", default=["averaging", "last-category", "markov1", "lstm", "cnn", "sasrec",
                                                      "dros-sasrec", "gbdt-seq", "gbdt-full", "logit-seq", "logit-full"])
    p.add_argument("--pairs", nargs="*", default=[], help="paired comparisons A:B (see above)")
    p.add_argument("--fixed-population", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args()

    files = predictions.inventory(args.predictions)
    if not files:
        raise SystemExit(f"no prediction files in {args.predictions}")
    ev = Evaluator({(name, length): predictions.read(path) for name, length, path in files},
                   args.fixed_population, max(args.lengths), args.n_bootstrap, args.seed)
    independent = sorted(n for n, length in ev.tables if length is None)

    metrics, confusion, tests = [], [], []
    for x in args.lengths:
        customers = ev.customers(x)
        scored = {n: ev.score(n, x, customers) for n in independent + sorted(n for n, l in ev.tables if l == x)}
        for name, s in scored.items():
            row = {"model": name, "length": x, "N": len(s.y)}
            for m in REPORTED:
                lo, hi = s.interval(m)
                row.update({m: s.point[m], f"{m} CI low": lo, f"{m} CI high": hi})
            metrics.append(row)
            k = config.NUM_CLASSES
            cm = np.bincount(s.y * (k + 1) + s.p, minlength=k * (k + 1)).reshape(k, k + 1)
            for t, counts in zip(config.CATEGORIES, cm):
                confusion.append({"model": name, "length": x, "true class": t,
                                  **dict(zip(config.CATEGORIES + (NOT_A_CATEGORY,), counts))})
        for llm_name in [n for n in args.llms if n in scored]:
            family = [{"length": x, "N": len(customers), "llm": llm_name, "baseline": b,
                       **evaluation.compare(scored[llm_name], scored[b], TESTED)}
                      for b in args.baselines if b in scored]
            if family:
                family = pd.DataFrame(family)
                family["McNemar p (Holm)"] = evaluation.holm(family["McNemar p"])
                for m in TESTED:
                    family[f"Diff {m} p (Holm)"] = evaluation.holm(family[f"Diff {m} p"])
                family["Holm family"] = (f"{llm_name} against {len(family)} baselines ({', '.join(family['baseline'])}) "
                                         f"at length {x}, one family per metric")
                tests.append(family)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(args.out / "confusion.csv", index=False)
    print(metrics[["model", "length", "N", "Acc", "F1 (w)", "Macro-F1", "F1 Clothing"]].round(3).to_string(index=False))
    if tests:
        significance = pd.concat(tests, ignore_index=True)
        significance.to_csv(args.out / "significance.csv", index=False)
        view = significance[["length", "llm", "baseline", "Diff F1 (w)", "Diff F1 (w) p (Holm)", "McNemar p (Holm)"]]
        print("\n" + view.round(4).to_string(index=False))
    if args.pairs:
        pairs = pd.DataFrame([row for spec in args.pairs for row in ev.pair(spec, args.lengths)])
        if len(pairs):
            pairs.to_csv(args.out / "pairs.csv", index=False)
            print("\n" + pairs[["pair", "length", "N", "Diff F1 (w)", "Diff F1 (w) CI low", "Diff F1 (w) CI high",
                                "Diff F1 (w) p", "McNemar p"]].round(4).to_string(index=False))
        else:
            print("\nno pair has prediction files for both sides in this folder")
    if all((name, config.TRAIN_LENGTH) in ev.tables for name in FACTORIAL.values()):
        factorial = ev.factorial(config.TRAIN_LENGTH)
        factorial.to_csv(args.out / "factorial.csv", index=False)
        print("\n" + factorial[factorial.metric == "F1 (w)"].round(4).to_string(index=False))
    print(f"\nwritten to {args.out}/")


class Evaluator:
    def __init__(self, tables, fixed_population, longest, n_bootstrap, seed):
        self.tables = tables
        self.fixed = population(tables, longest) if fixed_population else None
        self.n_bootstrap, self.rng, self.weights = n_bootstrap, np.random.default_rng(seed), {}

    def customers(self, *lengths):
        """Test customers scored at these lengths: present at all of them (or the fixed population)."""
        if self.fixed is not None:
            return self.fixed
        return sorted(set.intersection(*(set(population(self.tables, x)) for x in lengths)))

    def score(self, name, length, customers):
        df = self.tables.get((name, length))
        df = self.tables.get((name, None)) if df is None else df
        if df is None:
            raise SystemExit(f"no prediction file for {name} at length {length}")
        key = tuple(customers)
        if key not in self.weights:
            self.weights[key] = evaluation.bootstrap_weights(len(customers), self.n_bootstrap, self.rng)
        df = df.set_index("customer_id").loc[customers]
        pred = df["prediction"].map(CODE).fillna(evaluation.INVALID).astype(int).to_numpy()
        return evaluation.Scored(df["ground_truth"].map(CODE).to_numpy(), pred, self.weights[key])

    def pair(self, spec, lengths):
        (names_a, len_a), (names_b, len_b) = (side(s) for s in spec.split(":"))
        if len_a or len_b:
            runs = [(len_a or len_b, len_b or len_a)]
        else:
            runs = [(x, x) for x in lengths if all((n, x) in self.tables or (n, None) in self.tables
                                                    for n in names_a + names_b)]
        rows = []
        for xa, xb in runs:
            missing = [f"{n}@{x}" for names, x in ((names_a, xa), (names_b, xb)) for n in names
                       if (n, x) not in self.tables and (n, None) not in self.tables]
            if missing:
                print(f"pair {spec}: no prediction file for {', '.join(missing)}; skipped")
                continue
            customers = self.customers(xa, xb)
            a = evaluation.mean_of([self.score(n, xa, customers) for n in names_a])
            b = evaluation.mean_of([self.score(n, xb, customers) for n in names_b])
            rows.append({"pair": spec, "length": xa if xa == xb else f"{xa} vs {xb}", "N": len(customers),
                         **evaluation.compare(a, b, PAIR_METRICS)})
        return rows

    def factorial(self, length):
        """Main effect: mean of the four conditions with a field minus the four without; interactions alike."""
        customers = self.customers(length)
        scored = {k: self.score(name, length, customers) for k, name in FACTORIAL.items()}
        rows = []
        for metric in ["F1 (w)", "Acc", "Macro-F1", "F1 Clothing"]:
            for term in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
                sign = {k: np.prod([1 if k[i] else -1 for i in term]) for k in scored}
                point = sum(sign[k] * s.point[metric] for k, s in scored.items()) / 4
                draws = sum(sign[k] * s.draws[metric] for k, s in scored.items()) / 4
                lo, hi = np.percentile(draws, [2.5, 97.5])
                kind = "main effect" if len(term) == 1 else "interaction"
                rows.append({"metric": metric, "effect": f"{kind}: " + " x ".join(FACTORS[i] for i in term),
                             "estimate": point, "CI low": lo, "CI high": hi})
        return pd.DataFrame(rows)


def side(text):
    names, _, length = text.partition("@")
    return names.split(","), int(length) if length else None


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

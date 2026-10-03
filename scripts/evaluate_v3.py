"""The analyses of ANALYSIS_PLAN.md, on top of scripts/evaluate.py.

scripts/evaluate.py produces the per-split tables and the Holm-corrected tests of single models; point it at
one split folder at a time, e.g. --predictions results/predictions/bank_b_confirm. This script adds:

    summary     every model at every length: metrics, probability metrics (where files have p_ columns),
                and seed means +- SD for runs named <model> and <model>-seed<k>
    primary     the pre-registered test: macro-F1 at length 9 on bank_b_confirm, LLM seed mean against the
                non-LLM model with the best VALIDATION macro-F1 (so the baselines must be scored on val).
                Sensitivity analyses: --split bank_b_unfiltered, and --without-ties (customers whose target
                date holds two or more transactions are left out)
    families    the secondary tests on seed means: each LLM against each non-LLM comparator, per metric and
                length, paired bootstrap p, Holm over the comparators (the families of ANALYSIS_PLAN.md)
    shift       in-bank (bank_a_test) against cross-bank (bank_b_confirm) scores and the drop, per model
    subgroups   metrics by gender and by age band, from the split's demographics
    calibration reliability table and a cost-weighted Clothing operating point chosen on validation

    python scripts/evaluate_v3.py summary --split bank_b_confirm
    python scripts/evaluate_v3.py primary --llm qwen-full
    python scripts/evaluate_v3.py primary --llm qwen-full --split bank_b_unfiltered
    python scripts/evaluate_v3.py primary --llm qwen-full --without-ties
    python scripts/evaluate_v3.py families --llms qwen-full qwen-seq mistral-full
    python scripts/evaluate_v3.py shift
    python scripts/evaluate_v3.py subgroups --split bank_b_confirm --models qwen-full gbdt-full
    python scripts/evaluate_v3.py calibration --model qwen-full --cost-fn 5 --cost-fp 1
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, data, evaluation, predictions

CODE = {c: i for i, c in enumerate(config.CATEGORIES)}
SEED = re.compile(r"^(?P<base>.+?)-seed(?P<seed>\d+)$")
METRICS = ["Acc", "F1 (w)", "Macro-F1", "Bal. acc"] + evaluation.CLASS_F1
NON_LLM = ["averaging", "last-category", "markov1", "lstm", "cnn", "sasrec", "dros-sasrec",
           "gbdt-seq", "gbdt-full", "logit-seq", "logit-full"]


def load(folder: Path):
    """{(name, length): DataFrame} for one split folder."""
    return {(n, x): predictions.read(path) for n, x, path in predictions.inventory(folder)}


def codes(df):
    return (df["ground_truth"].map(CODE).to_numpy(),
            df["prediction"].map(CODE).fillna(evaluation.INVALID).astype(int).to_numpy())


def probs(df):
    cols = predictions.PROB_COLUMNS
    return df[cols].to_numpy(float) if all(c in df.columns for c in cols) else None


def family(tables, base, length):
    """All seeds of one model at one length: base, base-seed2, ..."""
    out = []
    for (n, x), df in tables.items():
        m = SEED.match(n)
        if x == length and (n == base or (m and m.group("base") == base)):
            out.append(df)
    return out


def summary(args):
    tables = load(args.predictions / args.split)
    rows = []
    for (n, x), df in sorted(tables.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        y, p = codes(df)
        row = {"model": n, "length": x, "N": len(df), **evaluation.point_metrics(y, p)}
        P = probs(df)
        if P is not None:
            row.update(evaluation.probability_metrics(y, P))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["length"] = table["length"].astype("Int64")            # empty for the length-independent baselines
    table["base"] = [SEED.match(n).group("base") if SEED.match(n) else n for n in table["model"]]
    numeric = [c for c in table.columns if c not in ("model", "base", "length", "N")]
    groups = table.groupby(["base", "length"], dropna=False)
    seeds = groups[numeric].mean().add_suffix(" mean").join(groups[numeric].std().add_suffix(" sd"))
    seeds = seeds[[f"{m} {k}" for m in numeric for k in ("mean", "sd")]]
    seeds.insert(0, "N", groups["N"].first())
    seeds.insert(0, "seeds", groups.size())
    seeds = seeds.reset_index().rename(columns={"base": "model"})
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns="base").to_csv(out / "summary.csv", index=False)
    seeds.to_csv(out / "summary_seed_means.csv", index=False)       # one row per model and length; sd is empty for one run
    view = table[["model", "length", "N", "Acc", "F1 (w)", "Macro-F1", "F1 Clothing"]
                 + [c for c in ("PR-AUC (macro)", "ECE") if c in table.columns]]
    print(view.round(3).to_string(index=False))
    print(f"\nwritten to {out}/summary.csv and summary_seed_means.csv")


def scored(dfs, customers, weights):
    """Seed mean (or single model) as an evaluation.Scored-like object on fixed customers."""
    items = []
    for df in dfs:
        df = df.set_index("customer_id").loc[customers].reset_index()
        y, p = codes(df)
        items.append(evaluation.Scored(y, p, weights))
    return evaluation.mean_of(items)


def primary(args):
    x = config.TRAIN_LENGTH
    val = load(args.predictions / "val")
    val_scores = {}
    for base in args.non_llm:
        dfs = family(val, base, x)
        if dfs:
            val_scores[base] = float(np.mean([evaluation.point_metrics(*codes(d))["Macro-F1"] for d in dfs]))
    if not val_scores:
        raise SystemExit("no non-LLM predictions on the val split: run the baselines with --splits val ...")
    comparator = max(val_scores, key=val_scores.get)
    print("validation macro-F1 of the non-LLM models:", {k: round(v, 4) for k, v in sorted(val_scores.items(), key=lambda kv: -kv[1])})
    print(f"comparator (best on validation): {comparator}")

    confirm = load(args.predictions / args.split)
    a, b = family(confirm, args.llm, x), family(confirm, comparator, x)
    if not a or not b:
        raise SystemExit(f"missing {args.split} predictions for the LLM or the comparator")
    customers = set.intersection(*(set(d["customer_id"]) for d in a + b))
    label, suffix = args.split, "" if args.split == "bank_b_confirm" else f"_{args.split}"
    if args.without_ties:
        tx, _ = data.load_split(args.data, args.split)
        tied = set(data.tie_report(tx)["tied_ids"].tolist())
        print(f"sensitivity analysis: {len(customers & tied):,} customers with two or more transactions on the "
              f"target date are left out")
        customers, label, suffix = customers - tied, f"{label} without tied target dates", f"{suffix}_without_ties"
    customers = sorted(customers)
    weights = evaluation.bootstrap_weights(len(customers), config.N_BOOTSTRAP, np.random.default_rng(config.SEED))
    res = evaluation.compare(scored(a, customers, weights), scored(b, customers, weights), ["Macro-F1"])
    kind = "PRIMARY ENDPOINT" if not suffix else "SENSITIVITY ANALYSIS of the primary endpoint"
    print(f"\n{kind}: macro-F1 at length {x}, {label}, N = {len(customers):,}")
    print(f"  {args.llm} ({len(a)} seeds) minus {comparator} ({len(b)} seeds): {res['Diff Macro-F1']:+.4f} "
          f"[{res['Diff Macro-F1 CI low']:+.4f}, {res['Diff Macro-F1 CI high']:+.4f}], bootstrap p = {res['Diff Macro-F1 p']:.4f}, "
          f"MDE = {res['Diff Macro-F1 MDE']:.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"llm": args.llm, "llm seeds": len(a), "comparator": comparator, "comparator seeds": len(b),
                   "population": label, "N": len(customers), **res}]).to_csv(
        args.out / f"primary_endpoint{suffix}.csv", index=False)


def families(args):
    tables = load(args.predictions / args.split)
    metrics = ["Acc", "F1 (w)", "Macro-F1", "Bal. acc"] + evaluation.CLASS_F1
    rows = []
    for x in config.EVAL_LENGTHS:
        present = [b for b in args.non_llm if family(tables, b, x)]
        for llm_name in args.llms:
            a = family(tables, llm_name, x)
            if not a or not present:
                continue
            customers = sorted(set.intersection(*(set(d["customer_id"]) for b in present for d in family(tables, b, x)),
                                                *(set(d["customer_id"]) for d in a)))
            weights = evaluation.bootstrap_weights(len(customers), args.n_bootstrap, np.random.default_rng(config.SEED))
            A = scored(a, customers, weights)
            fam = []
            for b in present:
                res = evaluation.compare(A, scored(family(tables, b, x), customers, weights), metrics)
                fam.append({"length": x, "N": len(customers), "llm": llm_name, "llm seeds": len(a), "comparator": b,
                            "comparator seeds": len(family(tables, b, x)), **res})
            fam = pd.DataFrame(fam)
            for m in metrics:
                fam[f"Diff {m} p (Holm)"] = evaluation.holm(fam[f"Diff {m} p"])
            fam["Holm family"] = (f"{llm_name} against {len(present)} non-LLM comparators ({', '.join(present)}) "
                                  f"at length {x} on {args.split}, one family per metric")
            rows.append(fam)
    table = pd.concat(rows, ignore_index=True)
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "significance_seed_means.csv", index=False)
    view = table[["length", "llm", "comparator", "Diff Macro-F1", "Diff Macro-F1 p (Holm)", "Diff F1 (w)", "Diff F1 (w) p (Holm)"]]
    print(view.round(4).to_string(index=False))
    print(f"\nwritten to {out}/significance_seed_means.csv")


def shift(args):
    """Seed means of every model scored on both splits at the training length."""
    x = config.TRAIN_LENGTH
    a, b = load(args.predictions / "bank_a_test"), load(args.predictions / "bank_b_confirm")
    bases = sorted({SEED.match(n).group("base") if SEED.match(n) else n for n, length in a if length == x})
    rows = []
    for base in bases:
        fa, fb = family(a, base, x), family(b, base, x)
        if not fa or not fb:
            continue
        ma = pd.DataFrame([evaluation.point_metrics(*codes(d)) for d in fa]).mean()
        mb = pd.DataFrame([evaluation.point_metrics(*codes(d)) for d in fb]).mean()
        row = {"model": base, "seeds in-bank": len(fa), "seeds cross-bank": len(fb)}
        for m in ["Acc", "F1 (w)", "Macro-F1"] + evaluation.CLASS_F1:
            row[f"{m} in-bank"], row[f"{m} cross-bank"], row[f"{m} drop"] = ma[m], mb[m], ma[m] - mb[m]
        rows.append(row)
    if not rows:
        raise SystemExit("no model has predictions on both bank_a_test and bank_b_confirm")
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "shift_in_vs_cross_bank.csv", index=False)
    view = table[["model"] + [f"{m} {k}" for m in ("F1 (w)", "Macro-F1") for k in ("in-bank", "cross-bank", "drop")]]
    print(view.round(3).to_string(index=False))


def subgroups(args):
    tables = load(args.predictions / args.split)
    _, demo = data.load_split(args.data, args.split)
    if demo is None:
        raise SystemExit("this split has no demographics")
    band = pd.cut(demo["age"], [0, 29, 39, 49, 59, 200], labels=["<30", "30-39", "40-49", "50-59", "60+"])
    rows = []
    for name in args.models:
        df = tables.get((name, config.TRAIN_LENGTH))
        if df is None:
            continue
        for group, values in (("gender", demo["gender"]), ("age band", band)):
            key = values.reindex(df["customer_id"]).to_numpy()
            for level in pd.unique(key):
                mask = key == level
                y, p = codes(df[mask])
                m = evaluation.point_metrics(y, p)
                rows.append({"model": name, "group": group, "level": level, "N": int(mask.sum()),
                             "Acc": m["Acc"], "Macro-F1": m["Macro-F1"], "R Clothing": m["R Clothing"],
                             "F1 Clothing": m["F1 Clothing"]})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / f"subgroups_{args.split}.csv", index=False)
    print(table.round(3).to_string(index=False))


def calibration(args):
    x = config.TRAIN_LENGTH
    val, test = load(args.predictions / "val"), load(args.predictions / args.split)
    if (args.model, x) not in test:
        raise SystemExit(f"no {args.model} predictions at length {x} in {args.split}")
    df = test[(args.model, x)]
    y, _ = codes(df)
    P = probs(df)
    rel = evaluation.reliability_table(y, P)
    args.out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(args.out / f"reliability_{args.model}_{args.split}.csv", index=False)
    print(rel.round(3).to_string(index=False))
    cls = CODE["Clothing"]
    if (args.model, x) in val and probs(val[(args.model, x)]) is not None:
        vy, _ = codes(val[(args.model, x)])
        op = evaluation.cost_weighted_threshold(vy, probs(val[(args.model, x)])[:, cls], cls, args.cost_fn, args.cost_fp)
        flag = P[:, cls] >= op["threshold"]
        tp = (flag & (y == cls)).sum()
        print(f"\nClothing operating point chosen on validation (cost FN {args.cost_fn}, FP {args.cost_fp}): "
              f"threshold {op['threshold']:.3f}; on {args.split}: recall {tp / max((y == cls).sum(), 1):.3f}, "
              f"precision {tp / max(flag.sum(), 1):.3f}")
    else:
        print("\n(no validation predictions with probabilities: score the model on --splits val to get the operating point)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/metrics_v3"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("--split", default="bank_b_confirm")
    q = sub.add_parser("primary"); q.add_argument("--llm", default="qwen-full")
    q.add_argument("--non-llm", nargs="+", default=NON_LLM)
    q.add_argument("--split", default="bank_b_confirm", help="bank_b_unfiltered for the sensitivity analysis")
    q.add_argument("--without-ties", action="store_true",
                   help="sensitivity analysis: leave out customers with two or more transactions on the target date")
    f = sub.add_parser("families"); f.add_argument("--split", default="bank_b_confirm")
    f.add_argument("--llms", nargs="+", default=["qwen-full", "qwen-seq", "mistral-full"])
    f.add_argument("--non-llm", nargs="+", default=NON_LLM)
    f.add_argument("--n-bootstrap", type=int, default=config.N_BOOTSTRAP)
    sub.add_parser("shift")
    g = sub.add_parser("subgroups"); g.add_argument("--split", default="bank_b_confirm")
    g.add_argument("--models", nargs="+", default=["qwen-full", "gbdt-full"])
    c = sub.add_parser("calibration"); c.add_argument("--model", default="qwen-full")
    c.add_argument("--split", default="bank_b_confirm")
    c.add_argument("--cost-fn", type=float, default=5.0); c.add_argument("--cost-fp", type=float, default=1.0)
    args = p.parse_args()
    {"summary": summary, "primary": primary, "families": families, "shift": shift, "subgroups": subgroups, "calibration": calibration}[args.cmd](args)


if __name__ == "__main__":
    main()

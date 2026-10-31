"""Calibration of every class, not only of the most probable one (Section 4.4 and Supporting Information).

The ECE of evaluation.probability_metrics bins the probability of the predicted class (top-label ECE). This
script adds, for every model with class probabilities, at length 9:
    * the one-versus-rest ECE of each class: the probability of class k against the share of targets that are k,
      in 15 equal-width bins (the binning of evaluation.probability_metrics), and their mean over the classes;
    * the Murphy decomposition of the multiclass Brier score, summed over the classes:
      Brier = reliability - resolution + uncertainty + a within-bin term (small; reported as the remainder);
    * the mean predicted probability and the observed share of each class.
Seed means where a model has several seeds. It also writes the one-versus-rest reliability table of Clothing for
the seed-42 run of a few models (Figure 4b). Descriptive: no tests.

    python scripts/calibration_classes.py --predictions results/predictions --out results/local/calibration
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, predictions

SEED = re.compile(r"^(?P<base>.+?)-seed(?P<seed>\d+)$")
CODE = {c: i for i, c in enumerate(config.CATEGORIES)}
PROBS = [f"p_{c}" for c in config.CATEGORIES]
N_BINS = 15
MODELS = ["markov1", "logit-seq", "logit-full", "gbdt-seq", "gbdt-full", "lstm", "cnn", "sasrec", "dros-sasrec",
          "mistral-full", "qwen-seq", "qwen-full"]


def runs_of(folder: Path, base: str, length: int) -> list[pd.DataFrame]:
    out = []
    for name, x, path in sorted(predictions.inventory(folder), key=lambda r: r[0]):
        m = SEED.match(name)
        if x == length and (name == base or (m and m.group("base") == base)):
            out.append(predictions.read(path))
    return out


def bins_of(p: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(p, np.linspace(0, 1, N_BINS + 1)[1:-1]), 0, N_BINS - 1)


def one_vs_rest(y: np.ndarray, p: np.ndarray) -> dict:
    """ECE and Brier decomposition of one class: p is the class probability, y is 1 when the target is the class."""
    b, base = bins_of(p), y.mean()
    ece = rel = res = 0.0
    for k in range(N_BINS):
        m = b == k
        if m.any():
            w = m.mean()
            ece += w * abs(y[m].mean() - p[m].mean())
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - base) ** 2
    return {"ECE": ece, "Brier": float(((p - y) ** 2).mean()), "reliability": rel, "resolution": res,
            "uncertainty": base * (1 - base), "mean probability": float(p.mean()), "observed share": float(base)}


def run_scores(df: pd.DataFrame) -> dict:
    y = df["ground_truth"].map(CODE).to_numpy()
    P = df[PROBS].to_numpy(dtype=float)
    per = {c: one_vs_rest((y == k).astype(float), P[:, k]) for k, c in enumerate(config.CATEGORIES)}
    out = {f"ECE {c}": per[c]["ECE"] for c in config.CATEGORIES}
    out["ECE (class mean)"] = float(np.mean([per[c]["ECE"] for c in config.CATEGORIES]))
    for part in ("Brier", "reliability", "resolution", "uncertainty"):
        out[part] = sum(per[c][part] for c in config.CATEGORIES)
    out["remainder"] = out["Brier"] - (out["reliability"] - out["resolution"] + out["uncertainty"])
    for c in config.CATEGORIES:
        out[f"mean p {c}"] = per[c]["mean probability"]
        out[f"share {c}"] = per[c]["observed share"]
    return out


def reliability_rows(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    y = (df["ground_truth"] == cls).to_numpy(dtype=float)
    p = df[f"p_{cls}"].to_numpy(dtype=float)
    b = bins_of(p)
    return pd.DataFrame([{"bin": k, "mean probability": p[b == k].mean(), "observed share": y[b == k].mean(),
                          "n": int((b == k).sum())} for k in range(N_BINS) if (b == k).any()])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--splits", nargs="+", default=["bank_a_test", "bank_b_confirm"])
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--reliability-models", nargs="+", default=["qwen-full", "dros-sasrec"])
    p.add_argument("--out", type=Path, default=Path("results/local/calibration"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in args.splits:
        for model in args.models:
            runs = [r for r in runs_of(args.predictions / split, model, config.TRAIN_LENGTH) if r[PROBS].notna().all().all()]
            if not runs:
                continue
            mean = pd.DataFrame([run_scores(r) for r in runs]).mean()
            rows.append({"split": split, "model": model, "seeds": len(runs), **mean.to_dict()})
        for model in args.reliability_models:
            seed42 = runs_of(args.predictions / split, model, config.TRAIN_LENGTH)[0]  # sorts first: no -seed suffix
            reliability_rows(seed42, "Clothing").to_csv(args.out / f"reliability_clothing_{model}_{split}.csv", index=False)
    table = pd.DataFrame(rows)
    assert (table["remainder"].abs() < 0.005).all(), "Brier decomposition does not add up"
    table.to_csv(args.out / "calibration_classes.csv", index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

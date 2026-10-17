"""Computational cost of the trained models, from the training logs and the scoring logs (Table 12).

LLM runs: wall-clock time and peak GPU memory of training (runs/<name>/training_log.json). Label scoring:
seconds per split and length (results/predictions/<name>.scoring.json); the scorers handle one prompt at a
time, so seconds / N is the time per customer, not a batched throughput. Baselines: seconds of the selected
configuration per seed and of the whole grid. Values the logs do not hold are left empty.

    python scripts/cost_table.py --runs runs --predictions results/predictions --out results/local/cost
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = re.compile(r"^(?P<base>.+?)-seed(?P<seed>\d+)$")
PASSES = {"first": 1, "sequence": 4}        # forward passes per prompt: first-token path, or one per answer


def base_of(name: str) -> str:
    m = SEED.match(name)
    return m.group("base") if m else name


def llm_training(runs: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(runs.glob("*/training_log.json")):
        j = json.loads(log.read_text())
        if "wall_clock_s" in j:
            rows.append({"run": log.parent.name, "model": base_of(log.parent.name), "base model": j["base_model"],
                         "hours": j["wall_clock_s"] / 3600, "peak GB": j.get("peak_gpu_memory_gb"),
                         "steps": j.get("steps"), "samples": j.get("n_samples"), "8-bit base": j.get("load_in_8bit")})
    return pd.DataFrame(rows)


def baseline_training(runs: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(runs.glob("*/training_log.json")):
        j = json.loads(log.read_text())
        if "selected" in j:
            seeds = j.get("seeds", {})
            per_seed = [v.get("seconds") for v in seeds.values() if isinstance(v, dict) and v.get("seconds") is not None]
            rows.append({"model": log.parent.name, "grid configurations": len(j.get("grid", [])),
                         "grid seconds": sum(g.get("seconds", 0) for g in j.get("grid", [])),
                         "seeds": len(seeds), "seconds per seed": float(np.mean(per_seed)) if per_seed else np.nan})
    return pd.DataFrame(rows)


def scoring(predictions: Path, split: str, length: int) -> pd.DataFrame:
    rows = []
    for log in sorted(predictions.glob("*.scoring.json")):
        if log.name.endswith("-gen.scoring.json"):
            continue
        for entry in json.loads(log.read_text()):
            for r in entry.get("runs", []):
                if r.get("split") == split and r.get("length") == length and "seconds" in r:
                    rows.append({"run": entry["name"], "model": base_of(entry["name"]), "scorer": entry["score"],
                                 "few-shot": entry.get("few_shot", False), "passes per prompt": PASSES.get(entry["score"]),
                                 "N": r["N"], "seconds": r["seconds"], "ms per customer": 1000 * r["seconds"] / r["N"]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--split", default="bank_b_confirm")
    p.add_argument("--length", type=int, default=9)
    p.add_argument("--out", type=Path, default=Path("results/local/cost"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = llm_training(args.runs)
    train.to_csv(args.out / "llm_training_runs.csv", index=False)
    g = train.groupby("model")
    llm = pd.DataFrame({"runs": g.size(), "hours mean": g["hours"].mean(), "hours min": g["hours"].min(),
                        "hours max": g["hours"].max(), "peak GB": g["peak GB"].max(), "steps": g["steps"].first()})
    score = scoring(args.predictions, args.split, args.length)
    score.to_csv(args.out / "scoring_runs.csv", index=False)
    s = score.groupby("model").agg(**{"scored runs": ("run", "size"), "N": ("N", "first"),
                                      "seconds mean": ("seconds", "mean"), "ms per customer": ("ms per customer", "mean"),
                                      "scorer": ("scorer", "first"), "passes per prompt": ("passes per prompt", "first")})
    table = llm.join(s, how="outer")
    table.index.name = "model"
    table.round(2).to_csv(args.out / "cost.csv")
    base = baseline_training(args.runs)
    base.round(1).to_csv(args.out / "baselines.csv", index=False)
    totals = pd.DataFrame([{"LoRA runs": len(train), "GPU hours, all LoRA runs": train["hours"].sum()}])
    totals.round(1).to_csv(args.out / "totals.csv", index=False)
    print(table.round(2).to_string())
    print(base.round(1).to_string(index=False))
    print(totals.round(1).to_string(index=False))


if __name__ == "__main__":
    main()

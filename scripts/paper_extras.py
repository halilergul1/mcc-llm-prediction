"""Small tables the manuscript quotes that the evaluation scripts only print or keep in JSON.

Writes CSV files so that scripts/trace_numbers.py can find every number:
    splits.csv           customers per split, at length 9 and length 14 (data/processed)
    filter_counts.csv    single-category input windows at lengths 4, 7, 9 and ties on the target date, per split
    calibration.csv      the cost-weighted Clothing operating point of one model: threshold chosen on validation,
                         recall and precision on the test split, against argmax
    generation.csv       free generation on bank_b_dev: answers outside the label set and agreement with label scoring
    primary_scores.csv   macro-F1 of the LLM (seed mean) and of the comparator on each population of the primary analysis

    python scripts/paper_extras.py --predictions results/predictions --data data/processed --out results/local/extras
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcc_llm import config, data, evaluation, predictions

CODE = {c: i for i, c in enumerate(config.CATEGORIES)}
SPLITS = ["train", "val", "bank_a_test", "bank_b_dev", "bank_b_confirm", "bank_b_unfiltered"]


def splits_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        tx, demo = data.load_split(data_dir, split)
        counts = tx.groupby("customer_id").size()
        rows.append({"split": split, "customers": int(len(counts)),
                     "at length 9": int((counts >= 10).sum()), "at length 14": int((counts >= 15).sum())})
    return pd.DataFrame(rows)


def filter_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        tx, _ = data.load_split(data_dir, split)
        row = {"split": split}
        for x in (4, 7, 9):
            s = data.single_category_share(data.last_windows(tx, x))
            row[f"single-category inputs at length {x}"] = s["single_category"]
            row[f"single-category inputs at length {x} (%)"] = round(100 * s["share"], 1)
        t = data.tie_report(tx)
        row.update({"customers": t["customers"], "tied target date": t["tied_last_date"],
                    "tied target date (%)": round(100 * t["tied_share"], 1),
                    "tied with two or more classes": t["tied_with_different_classes"]})
        rows.append(row)
    return pd.DataFrame(rows)


def read(folder: Path, name: str, length: int = config.TRAIN_LENGTH) -> pd.DataFrame:
    return predictions.read(folder / predictions.file_name(name, length))


def operating_point(pred_dir: Path, model: str, split: str, cost_fn: float, cost_fp: float) -> dict:
    cls = CODE["Clothing"]
    val, test = read(pred_dir / "val", model), read(pred_dir / split, model)
    cols = predictions.PROB_COLUMNS
    op = evaluation.cost_weighted_threshold(val["ground_truth"].map(CODE).to_numpy(), val[cols].to_numpy(float)[:, cls],
                                            cls, cost_fn, cost_fp)
    y = test["ground_truth"].map(CODE).to_numpy()
    flag = test[cols].to_numpy(float)[:, cls] >= op["threshold"]
    argmax = test["prediction"].map(CODE).to_numpy() == cls
    pos = y == cls
    return {"model": model, "split": split, "cost FN": cost_fn, "cost FP": cost_fp, "threshold": round(op["threshold"], 3),
            "recall": (flag & pos).sum() / pos.sum(), "precision": (flag & pos).sum() / flag.sum(),
            "flagged (%)": 100 * flag.mean(), "Clothing targets (%)": 100 * pos.mean(),
            "argmax recall": (argmax & pos).sum() / pos.sum(), "argmax precision": (argmax & pos).sum() / argmax.sum(),
            "argmax flagged (%)": 100 * argmax.mean()}


def macro_f1(df: pd.DataFrame, ids) -> float:
    d = df.set_index("customer_id").loc[ids]
    y = d["ground_truth"].map(CODE).to_numpy()
    pr = d["prediction"].map(CODE).fillna(evaluation.INVALID).astype(int).to_numpy()
    return evaluation.point_metrics(y, pr)["Macro-F1"]


def primary_scores(pred_dir: Path, data_dir: Path, llm: str, comparator: str) -> pd.DataFrame:
    """Scores behind the primary comparison and its two sensitivity analyses (evaluate_v3.py primary)."""
    seeds = [llm, f"{llm}-seed2", f"{llm}-seed3"]
    rows = []
    for split, without_ties in (("bank_b_confirm", False), ("bank_b_unfiltered", False), ("bank_b_confirm", True)):
        runs = [read(pred_dir / split, s) for s in seeds if (pred_dir / split / predictions.file_name(s, 9)).exists()]
        ids = sorted(set.intersection(*(set(r["customer_id"]) for r in runs)))
        if without_ties:
            tx, _ = data.load_split(data_dir, split)
            ids = sorted(set(ids) - set(data.tie_report(tx)["tied_ids"].tolist()))
        comp = read(pred_dir / split, comparator)
        rows.append({"population": split + (" without tied target dates" if without_ties else ""), "N": len(ids),
                     "llm": llm, "llm seeds": len(runs), "llm Macro-F1": float(np.mean([macro_f1(r, ids) for r in runs])),
                     "comparator": comparator, "comparator Macro-F1": macro_f1(comp, ids)})
    return pd.DataFrame(rows)


def generation_table(pred_dir: Path) -> pd.DataFrame:
    rows = []
    for log in sorted(pred_dir.glob("*-gen.scoring.json")):
        name = log.name.replace(".scoring.json", "")
        base = name[:-len("-gen")]
        for entry in json.loads(log.read_text()):
            labels = entry.get("labels", list(config.CATEGORIES))
            for r in entry.get("runs", []):
                gen = read(pred_dir / r["split"], name, r["length"]).set_index("customer_id")
                scored = read(pred_dir / r["split"], base, r["length"]).set_index("customer_id").loc[gen.index]
                rows.append({"model": base, "split": r["split"], "length": r["length"], "N": r["N"],
                             "not a category": r.get("not_a_category"),
                             "agreement with label scoring (%)": 100 * float((gen["prediction"] == scored["prediction"]).mean()),
                             "labels": "/".join(labels)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--models", nargs="+", default=["qwen-full"], help="models for the operating point")
    p.add_argument("--split", default="bank_b_confirm")
    p.add_argument("--cost-fn", type=float, default=5.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("results/local/extras"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in [("splits", splits_table(args.data)), ("filter_counts", filter_table(args.data)),
                        ("calibration", pd.DataFrame([operating_point(args.predictions, m, args.split, args.cost_fn, args.cost_fp)
                                                      for m in args.models])),
                        ("generation", generation_table(args.predictions)),
                        ("primary_scores", primary_scores(args.predictions, args.data, "qwen-full", "gbdt-full"))]:
        table.to_csv(args.out / f"{name}.csv", index=False)
        print(f"--- {name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()

"""Feature-based baselines: LightGBM and multinomial logistic regression, on two feature sets (Section 3.6).

    seq   what the sequence-only LLM sees: features of the category sequence
    full  what the full-context LLM sees: seq plus amounts, dates and demographics

Each model is selected on the validation split by config.SELECTION_METRIC over a small grid. Both learners
are deterministic as configured (LightGBM without row or feature subsampling), so each is fitted once and
there are no seed replicates. The fitted models are saved in --run-dir. Outputs, per split and length:
gbdt-seq, gbdt-full, logit-seq, logit-full (with class probabilities).

    python scripts/train_features.py
After the freeze, score the confirmation splits from the saved models:
    python scripts/train_features.py --score-only --splits bank_b_confirm bank_b_unfiltered
"""
import argparse
import json
from pathlib import Path

import joblib

from mcc_llm import config, data, evaluation, features, predictions      # no PyTorch here: LightGBM and
#                                                                         PyTorch each bring an OpenMP runtime

KINDS = ("seq", "full")


def feature_table(kind, windows, demo):
    return features.sequence_features(windows.codes()[:, :-1]) if kind == "seq" else features.full_features(windows, demo)


def fit(kind, args):
    """Grid search on the validation split; returns the fitted models and what is needed to encode new tables."""
    train_tx, train_demo = data.load_split(args.data, "train")
    val_tx, val_demo = data.load_split(args.data, "val")
    w_tr, w_va = data.last_windows(train_tx, config.TRAIN_LENGTH), data.last_windows(val_tx, config.TRAIN_LENGTH)
    y_tr, y_va = w_tr.codes()[:, -1], w_va.codes()[:, -1]
    X_tr, X_va = feature_table(kind, w_tr, train_demo), feature_table(kind, w_va, val_demo)

    scored = []
    for params in features.LGBM_GRID:
        m = features.fit_lightgbm(X_tr, y_tr, X_va, y_va, params, seed=args.seed)
        scored.append((evaluation.selection_score(y_va, m.predict(X_va)), params, m))
    gbdt_score, gbdt_params, gbdt = max(scored, key=lambda t: t[0])

    cats, columns = features.dummy_columns(X_tr)
    oh_tr, oh_va = features.one_hot(X_tr, cats, columns), features.one_hot(X_va, cats, columns)
    scored = [(evaluation.selection_score(y_va, features.fit_logistic(oh_tr, y_tr, **g).predict(oh_va)), g)
              for g in features.LOGIT_GRID]
    logit_score, logit_params = max(scored, key=lambda t: t[0])
    logit = features.fit_logistic(oh_tr, y_tr, **logit_params)

    log = {"gbdt": {"params": gbdt_params, "val": gbdt_score, "best_iteration": int(gbdt.best_iteration_ or 0)},
           "logit": {"params": logit_params, "val": logit_score}, "selection_metric": config.SELECTION_METRIC}
    print(f"{kind}: gbdt {gbdt_params} val {config.SELECTION_METRIC} {gbdt_score:.4f}; "
          f"logit {logit_params} val {logit_score:.4f}", flush=True)
    return {"gbdt": gbdt, "logit": logit, "categoricals": cats, "columns": columns,
            "categories": list(config.CATEGORIES)}, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/predictions"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/features"))
    p.add_argument("--splits", nargs="+", default=list(config.PRE_SPLITS),
                   help="evaluation splits to score (default: the splits that may be scored before the freeze)")
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--score-only", action="store_true",
                   help="train nothing: load the saved models of --run-dir and score --splits")
    args = p.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = {}
    for kind in KINDS:
        path = args.run_dir / f"{kind}.joblib"
        if args.score_only:
            if not path.exists():
                raise SystemExit(f"{path} not found: train the models first (this script without --score-only)")
            bundle = joblib.load(path)
            if bundle["categories"] != list(config.CATEGORIES):
                raise SystemExit(f"{path} was trained with the classes {bundle['categories']}")
        else:
            bundle, log[kind] = fit(kind, args)
            joblib.dump(bundle, path)
        for split in args.splits:
            tx, demo = data.load_split(args.data, split)
            for x in args.lengths:
                w = data.last_windows(tx, x)
                if len(w) == 0:
                    continue
                X = feature_table(kind, w, demo)
                for name, proba in ((f"gbdt-{kind}", bundle["gbdt"].predict_proba(X)),
                                    (f"logit-{kind}", bundle["logit"].predict_proba(
                                        features.one_hot(X, bundle["categoricals"], bundle["columns"])))):
                    predictions.write(args.out / split / predictions.file_name(name, x), w.customer_id, w.target,
                                      predictions.to_names(proba.argmax(1)), probs=proba)
            print(f"  {kind}: {split} written", flush=True)
    if log:
        (args.run_dir / "selection.json").write_text(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()

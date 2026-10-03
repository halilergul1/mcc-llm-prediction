"""Train the DROS-SASRec baseline on the training bank and predict the evaluation splits (Section 3.6).

The SASRec backbone of the official DROS code, trained with its distributionally robust
objective: binary cross-entropy on the target and one sampled negative class, plus alpha
times the DRO term with temperature beta. The model reads the last 9 inputs
(config.DROS["state_size"]), also at length 14.

Grid search: every combination of the values given (default: config.DROS_GRID, alpha x beta) is
trained with the first seed, and the one with the best validation score (config.SELECTION_METRIC) is
kept and refitted with every other seed. One value per option trains one configuration,
e.g. --alpha 0.1 --beta 1.0.

Every model is saved in --run-dir and scored on every evaluation split. Without training, from the saved models:
    python scripts/train_dros.py --score-only --splits bank_b_confirm

The official code has no licence file and is not included here. Get it first:
    git clone https://github.com/YangZhengyi98/DROS third_party/DROS && git -C third_party/DROS checkout 44ff99d
"""
import argparse
from pathlib import Path

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    training.add_run_arguments(p, "runs/dros-sasrec")
    p.add_argument("--dros-dir", type=Path, default=Path("third_party/DROS"), help="clone of the official DROS code")
    p.add_argument("--alpha", type=float, nargs="+", default=list(config.DROS_GRID["alpha"]))
    p.add_argument("--beta", type=float, nargs="+", default=list(config.DROS_GRID["beta"]))
    p.add_argument("--hidden-size", type=int, nargs="+", default=[config.DROS["hidden_size"]])
    p.add_argument("--dropout", type=float, nargs="+", default=[config.DROS["dropout"]])
    p.add_argument("--lr", type=float, nargs="+", default=[config.DROS_TRAINING["lr"]])
    p.add_argument("--max-epochs", type=int, default=config.DROS_TRAINING["max_epochs"])
    args = p.parse_args()

    candidates = training.grid(alpha=args.alpha, beta=args.beta, hidden_size=args.hidden_size,
                               dropout=args.dropout, lr=args.lr)
    configs = [{**config.DROS, **config.DROS_TRAINING, "max_epochs": args.max_epochs, **c} for c in candidates]
    training.run("dros", "dros-sasrec", configs,
                 lambda cfg, train, val, seed: training.fit_dros(cfg, train, val, args.dros_dir, args.device, seed),
                 args, dros_dir=args.dros_dir)


if __name__ == "__main__":
    main()

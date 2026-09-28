"""Train the DROS-SASRec baseline on the training bank and predict the test customers (Section 3.6).

The SASRec backbone of the official DROS code, trained with its distributionally robust
objective: binary cross-entropy on the target and one sampled negative class, plus alpha
times the DRO term with temperature beta.

Grid search: every combination of the values given (default: config.DROS_GRID, alpha x beta) is
trained and the one with the best validation weighted F1 is kept. One value per option trains
one model, e.g. --alpha 0.1 --beta 1.0.

The official code has no licence file and is not included here. Get it first:
    git clone https://github.com/YangZhengyi98/DROS third_party/DROS && git -C third_party/DROS checkout 44ff99d
"""
import argparse
from pathlib import Path

import torch

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/dros-sasrec"))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--dros-dir", type=Path, default=Path("third_party/DROS"), help="clone of the official DROS code")
    p.add_argument("--alpha", type=float, nargs="+", default=list(config.DROS_GRID["alpha"]))
    p.add_argument("--beta", type=float, nargs="+", default=list(config.DROS_GRID["beta"]))
    p.add_argument("--hidden-size", type=int, nargs="+", default=[config.DROS["hidden_size"]])
    p.add_argument("--dropout", type=float, nargs="+", default=[config.DROS["dropout"]])
    p.add_argument("--lr", type=float, nargs="+", default=[config.DROS_TRAINING["lr"]])
    p.add_argument("--max-epochs", type=int, default=config.DROS_TRAINING["max_epochs"])
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train, val = training.training_windows(args.data)
    candidates = training.grid(alpha=args.alpha, beta=args.beta, hidden_size=args.hidden_size,
                               dropout=args.dropout, lr=args.lr)
    configs = [{**config.DROS, **config.DROS_TRAINING, "max_epochs": args.max_epochs, **c} for c in candidates]
    fitted, log = training.grid_search(
        "dros", configs, lambda cfg: training.fit_dros(cfg, train, val, args.dros_dir, args.device, args.seed),
        val, args.device)
    training.save_run(args.run_dir, "dros", fitted, log)
    print(f"selected {fitted.cfg}; test predictions:")
    training.write_test_predictions("dros", fitted, args.data, args.predictions, "dros-sasrec", device=args.device)


if __name__ == "__main__":
    main()

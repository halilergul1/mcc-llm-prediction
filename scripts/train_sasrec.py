"""Train the SASRec baseline on the training bank and predict the test customers (Section 3.6).

Two self-attention blocks with point-wise feed-forward layers, learned positional embeddings
and causal masking (PyTorch port of SASRec). Training uses the one-step-shifted target
sequence, so every position predicts the next category and the loss is summed over positions;
a four-way linear classifier replaces the ranking head. Only the last position is used at
evaluation. Options that take several values run a grid search on the validation split.
"""
import argparse
from pathlib import Path

import torch

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/sasrec"))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--hidden-units", type=int, nargs="+", default=[config.SASREC["hidden_units"]])
    p.add_argument("--num-blocks", type=int, nargs="+", default=[config.SASREC["num_blocks"]])
    p.add_argument("--num-heads", type=int, nargs="+", default=[config.SASREC["num_heads"]])
    p.add_argument("--dropout", type=float, nargs="+", default=[config.SASREC["dropout_rate"]])
    p.add_argument("--lr", type=float, nargs="+", default=[config.SASREC_TRAINING["lr"]])
    p.add_argument("--batch-size", type=int, nargs="+", default=[config.SASREC_TRAINING["batch_size"]])
    p.add_argument("--max-epochs", type=int, default=config.SASREC_TRAINING["max_epochs"])
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train, val = training.training_windows(args.data)
    candidates = training.grid(hidden_units=args.hidden_units, num_blocks=args.num_blocks, num_heads=args.num_heads,
                               dropout_rate=args.dropout, lr=args.lr, batch_size=args.batch_size)
    configs = [{**config.SASREC, **config.SASREC_TRAINING, "max_epochs": args.max_epochs, **c} for c in candidates]
    fitted, log = training.grid_search(
        "sasrec", configs, lambda cfg: training.fit_sasrec(cfg, train, val, args.device, args.seed), val, args.device)
    training.save_run(args.run_dir, "sasrec", fitted, log)
    print(f"selected {fitted.cfg}; test predictions:")
    training.write_test_predictions("sasrec", fitted, args.data, args.predictions, "sasrec", device=args.device)


if __name__ == "__main__":
    main()

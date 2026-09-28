"""Train the LSTM baseline on the training bank and predict the test customers (Section 3.6).

Single-layer LSTM (hidden size 128) over the integer-encoded category sequence, class-weighted
cross-entropy, Adam, early stopping on validation loss. One sample per customer: the last
9 transactions are the input and the 10th, most recent one is the target.

Grid search: every combination of the values given (default: config.LSTM_GRID) is trained and
the one with the best validation weighted F1 is kept. One value per option trains one model:
    python scripts/train_lstm.py --hidden-size 128 --lr 1e-3
"""
import argparse
from pathlib import Path

import torch

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/lstm"))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--hidden-size", type=int, nargs="+", default=list(config.LSTM_GRID["hidden_size"]))
    p.add_argument("--lr", type=float, nargs="+", default=list(config.LSTM_GRID["lr"]))
    p.add_argument("--batch-size", type=int, nargs="+", default=[config.NN_TRAINING["batch_size"]])
    p.add_argument("--max-epochs", type=int, default=config.NN_TRAINING["max_epochs"])
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train, val = training.training_windows(args.data)
    configs = [{**config.NN_TRAINING, "max_epochs": args.max_epochs, **c}
               for c in training.grid(hidden_size=args.hidden_size, lr=args.lr, batch_size=args.batch_size)]
    fitted, log = training.grid_search(
        "lstm", configs, lambda cfg: training.fit_classifier("lstm", cfg, train, val, args.device, args.seed),
        val, args.device)
    training.save_run(args.run_dir, "lstm", fitted, log)
    print(f"selected {fitted.cfg}; test predictions:")
    training.write_test_predictions("lstm", fitted, args.data, args.predictions, "lstm", device=args.device)


if __name__ == "__main__":
    main()

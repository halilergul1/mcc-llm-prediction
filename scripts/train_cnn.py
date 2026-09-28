"""Train the CNN baseline on the training bank and predict the test customers (Section 3.6).

Two stacked convolutions (32 and 64 channels, kernel 3) with adaptive average pooling and a
linear head, class-weighted cross-entropy, Adam, early stopping on validation loss. The
pooling layer lets one model read inputs of any length. One sample per customer: the last
9 transactions are the input and the 10th, most recent one is the target.
Options that take several values run a grid search on the validation split (weighted F1).
"""
import argparse
from pathlib import Path

import torch

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/cnn"))
    p.add_argument("--predictions", type=Path, default=Path("results/predictions"))
    p.add_argument("--lr", type=float, nargs="+", default=[config.NN_TRAINING["lr"]])
    p.add_argument("--batch-size", type=int, nargs="+", default=[config.NN_TRAINING["batch_size"]])
    p.add_argument("--max-epochs", type=int, default=config.NN_TRAINING["max_epochs"])
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train, val = training.training_windows(args.data)
    configs = [{**config.CNN, **config.NN_TRAINING, "max_epochs": args.max_epochs, **c}
               for c in training.grid(lr=args.lr, batch_size=args.batch_size)]
    fitted, log = training.grid_search(
        "cnn", configs, lambda cfg: training.fit_classifier("cnn", cfg, train, val, args.device, args.seed),
        val, args.device)
    training.save_run(args.run_dir, "cnn", fitted, log)
    print(f"selected {fitted.cfg}; test predictions:")
    training.write_test_predictions("cnn", fitted, args.data, args.predictions, "cnn", device=args.device)


if __name__ == "__main__":
    main()

"""Train the LSTM baseline on the training bank and predict the evaluation splits (Section 3.6).

Single-layer LSTM (hidden size 128) over learned embeddings of the class codes, cross-entropy
(unweighted or weighted by inverse class frequency), Adam, early stopping on validation loss. One sample
per customer: the last 9 transactions are the input and the 10th, most recent one is the target.

Grid search: every combination of the values given (default: config.LSTM_GRID and both loss weightings)
is trained with the first seed, and the one with the best validation score (config.SELECTION_METRIC) is
kept and refitted with every other seed. One value per option trains one configuration:
    python scripts/train_lstm.py --hidden-size 128 --lr 1e-3 --class-weights none

Every model is saved in --run-dir and scored on every evaluation split. Without training, from the saved models:
    python scripts/train_lstm.py --score-only --splits bank_b_confirm
"""
import argparse

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    training.add_run_arguments(p, "runs/lstm")
    p.add_argument("--hidden-size", type=int, nargs="+", default=list(config.LSTM_GRID["hidden_size"]))
    p.add_argument("--lr", type=float, nargs="+", default=list(config.LSTM_GRID["lr"]))
    p.add_argument("--batch-size", type=int, nargs="+", default=[config.NN_TRAINING["batch_size"]])
    p.add_argument("--max-epochs", type=int, default=config.NN_TRAINING["max_epochs"])
    p.add_argument("--class-weights", nargs="+", default=list(config.CLASS_WEIGHTS), choices=["none", "inverse"])
    args = p.parse_args()

    configs = [{**config.LSTM, **config.NN_TRAINING, "max_epochs": args.max_epochs, **c}
               for c in training.grid(hidden_size=args.hidden_size, lr=args.lr, batch_size=args.batch_size,
                                      class_weights=args.class_weights)]
    training.run("lstm", "lstm", configs,
                 lambda cfg, train, val, seed: training.fit_classifier("lstm", cfg, train, val, args.device, seed), args)


if __name__ == "__main__":
    main()

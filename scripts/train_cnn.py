"""Train the CNN baseline on the training bank and predict the evaluation splits (Section 3.6).

Three causal dilated convolutions (kernel 3, dilations 1, 2 and 4) over learned embeddings of the class
codes; the output at the last position feeds a linear head, so one model reads inputs of any length.
Cross-entropy (unweighted or weighted by inverse class frequency), Adam, early stopping on validation
loss. One sample per customer: the last 9 transactions are the input and the 10th, most recent one is
the target.

Grid search: every combination of the values given (default: config.CNN_GRID and both loss weightings)
is trained with the first seed, and the one with the best validation score (config.SELECTION_METRIC) is
kept and refitted with every other seed. One value per option trains one configuration:
    python scripts/train_cnn.py --lr 1e-3 --batch-size 64 --class-weights none

Every model is saved in --run-dir. After the freeze, score the confirmation splits from the saved models:
    python scripts/train_cnn.py --score-only --splits bank_b_confirm bank_b_unfiltered
"""
import argparse

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    training.add_run_arguments(p, "runs/cnn")
    p.add_argument("--lr", type=float, nargs="+", default=list(config.CNN_GRID["lr"]))
    p.add_argument("--batch-size", type=int, nargs="+", default=list(config.CNN_GRID["batch_size"]))
    p.add_argument("--max-epochs", type=int, default=config.NN_TRAINING["max_epochs"])
    p.add_argument("--class-weights", nargs="+", default=list(config.CLASS_WEIGHTS), choices=["none", "inverse"])
    args = p.parse_args()

    configs = [{**config.CNN, **config.NN_TRAINING, "max_epochs": args.max_epochs, **c}
               for c in training.grid(lr=args.lr, batch_size=args.batch_size, class_weights=args.class_weights)]
    training.run("cnn", "cnn", configs,
                 lambda cfg, train, val, seed: training.fit_classifier("cnn", cfg, train, val, args.device, seed), args)


if __name__ == "__main__":
    main()

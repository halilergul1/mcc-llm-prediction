"""Train the SASRec baseline on the training bank and predict the evaluation splits (Section 3.6).

Two self-attention blocks with point-wise feed-forward layers, learned positional embeddings
and causal masking (PyTorch port of SASRec). Training uses the one-step-shifted target
sequence, so every position predicts the next category and the loss is averaged over positions;
a four-way linear classifier replaces the ranking head. Only the last position is used at
evaluation. The model reads the last 9 inputs (config.SASREC["maxlen"]), also at length 14.

Grid search: every combination of the values given (default: config.SASREC_GRID and both loss weightings)
is trained with the first seed, and the one with the best validation score (config.SELECTION_METRIC) is
kept and refitted with every other seed. One value per option trains one configuration:
    python scripts/train_sasrec.py --hidden-units 50 --dropout 0.2 --lr 1e-3 --class-weights none

Every model is saved in --run-dir and scored on every evaluation split. Without training, from the saved models:
    python scripts/train_sasrec.py --score-only --splits bank_b_confirm
"""
import argparse

from mcc_llm import config, training


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    training.add_run_arguments(p, "runs/sasrec")
    p.add_argument("--hidden-units", type=int, nargs="+", default=list(config.SASREC_GRID["hidden_units"]))
    p.add_argument("--num-blocks", type=int, nargs="+", default=[config.SASREC["num_blocks"]])
    p.add_argument("--num-heads", type=int, nargs="+", default=[config.SASREC["num_heads"]])
    p.add_argument("--dropout", type=float, nargs="+", default=list(config.SASREC_GRID["dropout_rate"]))
    p.add_argument("--lr", type=float, nargs="+", default=list(config.SASREC_GRID["lr"]))
    p.add_argument("--batch-size", type=int, nargs="+", default=[config.SASREC_TRAINING["batch_size"]])
    p.add_argument("--max-epochs", type=int, default=config.SASREC_TRAINING["max_epochs"])
    p.add_argument("--class-weights", nargs="+", default=list(config.CLASS_WEIGHTS), choices=["none", "inverse"])
    args = p.parse_args()

    candidates = training.grid(hidden_units=args.hidden_units, num_blocks=args.num_blocks, num_heads=args.num_heads,
                               dropout_rate=args.dropout, lr=args.lr, batch_size=args.batch_size,
                               class_weights=args.class_weights)
    configs = [{**config.SASREC, **config.SASREC_TRAINING, "max_epochs": args.max_epochs, **c} for c in candidates]
    training.run("sasrec", "sasrec", configs,
                 lambda cfg, train, val, seed: training.fit_sasrec(cfg, train, val, args.device, seed), args)


if __name__ == "__main__":
    main()

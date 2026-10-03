"""The shared driver of the baseline scripts: every seed is saved, and saved models score like fresh ones."""
import argparse

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mcc_llm import config, data, features, training  # noqa: E402


def make_splits(folder, rng):
    codes = [5411, 5541, 5651, 4812]
    for name, first in (("train", 100), ("val", 300), ("bank_b_dev", 400), ("bank_b_confirm", 500)):
        rows = [(cid, f"2014-07-{d:02d}", float(rng.integers(1, 50)), int(rng.choice(codes)))
                for cid in range(first, first + 40) for d in range(1, 17)]
        tx = pd.DataFrame(rows, columns=["customer_id", "date", "amount", "mcc"])
        tx["category"] = data.map_categories(tx["mcc"])
        data.save_split(folder, name, tx, None)


def arguments(folder, **changes):
    args = dict(data=folder / "processed", run_dir=folder / "runs" / "lstm", predictions=folder / "pred",
                splits=["val", "bank_b_dev"], seeds=[42, 2], score_only=False, device="cpu")
    return argparse.Namespace(**{**args, **changes})


def test_saved_models_score_like_fresh_ones(tmp_path):
    make_splits(tmp_path / "processed", np.random.default_rng(0))
    cfg = {**config.LSTM, **config.NN_TRAINING, "hidden_size": 8, "max_epochs": 2, "class_weights": "none"}
    fit = lambda c, train, val, seed: training.fit_classifier("lstm", c, train, val, "cpu", seed)   # noqa: E731
    training.run("lstm", "lstm", [cfg], fit, arguments(tmp_path))
    assert (tmp_path / "runs/lstm/model.pt").exists() and (tmp_path / "runs/lstm/model-seed2.pt").exists()
    assert not (tmp_path / "pred/bank_b_confirm").exists()                  # nothing scored that was not asked for

    training.run("lstm", "lstm", [cfg], fit, arguments(tmp_path, score_only=True, predictions=tmp_path / "again",
                                                       splits=["bank_b_dev", "bank_b_confirm"]))
    for name in ("lstm_last9.csv", "lstm-seed2_last9.csv"):
        fresh, saved = pd.read_csv(tmp_path / "pred/bank_b_dev" / name), pd.read_csv(tmp_path / "again/bank_b_dev" / name)
        pd.testing.assert_frame_equal(fresh, saved)
    assert (tmp_path / "again/bank_b_confirm/lstm-seed2_last14.csv").exists()


def test_padded_positions_do_not_change_sasrec():
    cfg = {**config.SASREC, "hidden_units": 8}
    net = training.build_model("sasrec", cfg).eval()
    short = np.array([[1, 2, 0, 3]])                                        # 4 inputs, left-padded to maxlen 9
    (tokens,) = training.make_inputs("sasrec", short, cfg)
    assert tokens.shape == (1, 9) and int((tokens == 0).sum()) == 5
    hidden = net.item_emb(tokens)                                           # padding embeds to zero
    assert torch.all(hidden[0, :5] == 0)
    assert training.forward_logits("sasrec", net, (tokens,)).shape == (1, 4)


def test_one_hot_uses_the_training_levels():
    train = pd.DataFrame({"x": [1.0, 2.0], "job": pd.Categorical(["retired", "self-employed"])})
    other = pd.DataFrame({"x": [3.0], "job": pd.Categorical(["student"])})         # a level the training data lacks
    cats, columns = features.dummy_columns(train)
    assert cats == ["job"] and columns == ["x", "job_retired", "job_self-employed"]
    encoded = features.one_hot(other, cats, columns)
    assert list(encoded.columns) == columns and encoded.iloc[0].tolist() == [3.0, 0.0, 0.0]

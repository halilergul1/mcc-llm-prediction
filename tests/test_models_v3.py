import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcc_llm import evaluation, features, models  # noqa: E402


def test_lstm_and_cnn_read_integer_codes_and_any_length():
    for arch, cfg in (("lstm", {"hidden_size": 16, "embedding_dim": 8}), ("cnn", {"embedding_dim": 8, "channels": 16})):
        net = models.build_model(arch, cfg)
        for x in (4, 9, 14):
            codes = np.random.default_rng(0).integers(0, 4, (5, x))
            (inp,) = models.make_inputs(arch, codes, cfg)
            assert inp.dtype == torch.long
            assert models.forward_logits(arch, net, (inp,)).shape == (5, 4)


def test_cnn_is_causal_at_the_last_position():
    net = models.build_model("cnn", {"embedding_dim": 8, "channels": 16}).eval()
    a = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0]])
    b = a.clone(); b[0, -1] = 3                              # change only the most recent input
    assert not torch.allclose(net(a), net(b))                # the last position sees the most recent input


def test_permuting_class_codes_is_not_an_order():
    # with embeddings, relabelling the classes is a pure permutation: no class is "between" two others
    net = models.build_model("lstm", {"hidden_size": 8, "embedding_dim": 4})
    assert net.embedding.num_embeddings == 4


def test_probability_metrics_perfect_model():
    y = np.array([0, 1, 2, 3] * 25)
    P = np.eye(4)[y]
    m = evaluation.probability_metrics(y, P)
    assert m["PR-AUC (macro)"] == 1.0 and m["Top-2 acc"] == 1.0 and m["ECE"] == 0.0 and m["Brier"] == 0.0


def test_sequence_features_use_inputs_only():
    codes = np.array([[0, 0, 1, 2], [3, 3, 3, 3]])
    f = features.sequence_features(codes)
    assert list(f["count_Clothing"]) == [2, 0] and list(f["last_is_Food and grocery"]) == [1, 0]
    assert list(f["since_Other"]) == [4, 0] and list(f["switches"]) == [2, 0]

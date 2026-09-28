import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from mcc_llm import evaluation


def test_point_metrics_equal_scikit_learn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    s = evaluation.Scored(y, p, evaluation.bootstrap_weights(500, 200, rng))
    assert np.isclose(s.point["Acc"], accuracy_score(y, p))
    assert np.isclose(s.point["F1 (w)"], f1_score(y, p, average="weighted"))
    assert np.isclose(s.point["Macro-F1"], f1_score(y, p, average="macro"))
    assert np.isclose(s.point["Bal. acc"], balanced_accuracy_score(y, p))


def test_holm():
    assert np.allclose(evaluation.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

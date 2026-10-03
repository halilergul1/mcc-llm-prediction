import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from mcc_llm import evaluation


def test_bootstrap_engine_equals_scikit_learn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    engine = evaluation.metrics_from_confusion(np.bincount(y * 4 + p, minlength=16).reshape(4, 4))
    point = evaluation.point_metrics(y, p)
    assert all(np.isclose(engine[m], point[m]) for m in point)
    assert np.isclose(point["Acc"], accuracy_score(y, p))
    assert np.isclose(point["F1 (w)"], f1_score(y, p, average="weighted"))
    assert np.isclose(point["Macro-F1"], f1_score(y, p, average="macro"))
    assert np.isclose(point["Bal. acc"], balanced_accuracy_score(y, p))


def test_answer_that_is_not_a_category_is_an_error():
    y = np.array([0, 1, 2, 3, 2])
    p = np.array([0, 1, 2, 3, evaluation.INVALID])
    point = evaluation.point_metrics(y, p)
    engine = evaluation.metrics_from_confusion(np.bincount(y * 5 + p, minlength=20).reshape(4, 5))
    assert np.isclose(point["Acc"], 0.8) and np.isclose(point["R Food and grocery"], 0.5) and np.isclose(point["P Food and grocery"], 1.0)
    assert all(np.isclose(engine[m], point[m]) for m in point)


def test_paired_comparison_and_holm():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 300)
    weights = evaluation.bootstrap_weights(300, 500, rng)
    same = evaluation.compare(evaluation.Scored(y, y, weights), evaluation.Scored(y, y, weights), ["F1 (w)"])
    assert same["Diff F1 (w)"] == 0 and same["McNemar p"] == 1.0
    assert np.allclose(evaluation.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])

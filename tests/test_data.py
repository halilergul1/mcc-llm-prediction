import numpy as np
import pandas as pd

from mcc_llm import baselines, data


def transactions():
    rows = [(1, f"2014-07-{d:02d}", 10.0, mcc) for d, mcc in enumerate([5411] * 5 + [5541] * 6, start=1)]  # 11 txns
    rows += [(2, f"2014-07-{d:02d}", 5.0, 5411) for d in range(1, 13)]                                  # one category
    rows += [(3, f"2014-07-{d:02d}", 5.0, 5651 if d % 2 else 4812) for d in range(1, 10)]               # 9 txns
    df = pd.DataFrame(rows, columns=["customer_id", "date", "amount", "mcc"])
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = data.map_categories(df["mcc"])
    return df


def test_mcc_map():
    mcc = pd.Series([5411, 5462, 5812, 5814, 5541, 5542, 5983, 5651, 5137, 5941, 4812])
    assert list(data.map_categories(mcc)) == ["Grocery", "Grocery", "Other", "Other", "Gas stations", "Gas stations",
                                              "Gas stations", "Clothing", "Clothing", "Clothing", "Other"]


def test_cohort_filters():
    assert list(data.eligible_customers(transactions())) == [1]


def test_x_plus_one_windows():
    w = data.last_windows(transactions(), 9)
    assert list(w.customer_id) == [1, 2]                      # customer 3 has only 9 transactions
    assert list(w.categories[0]) == ["Grocery"] * 4 + ["Gas stations"] * 6
    assert list(w.target) == ["Gas stations", "Grocery"]
    assert len(data.last_windows(transactions(), 10)) == 2 and len(data.last_windows(transactions(), 11)) == 1


def test_averaging_ties_go_to_lowest_code():
    codes = np.array([[2, 2, 1, 1], [3, 3, 3, 0]])
    assert list(baselines.averaging_predictions(codes)) == [1, 3]

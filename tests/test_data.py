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
    assert list(data.map_categories(mcc)) == ["Food and grocery", "Food and grocery", "Other", "Other", "Gas stations",
                                              "Gas stations", "Gas stations", "Clothing", "Clothing", "Clothing", "Other"]


def test_cohort_filters():
    assert list(data.eligible_customers(transactions())) == [1]
    assert list(data.eligible_customers(transactions(), on_inputs=False)) == [1]


def test_x_plus_one_windows():
    w = data.last_windows(transactions(), 9)
    assert list(w.customer_id) == [1, 2]                      # customer 3 has only 9 transactions
    assert list(w.categories[0]) == ["Food and grocery"] * 4 + ["Gas stations"] * 6
    assert list(w.target) == ["Gas stations", "Food and grocery"]
    assert len(data.last_windows(transactions(), 10)) == 2 and len(data.last_windows(transactions(), 11)) == 1


def test_filter_never_looks_at_the_target():
    # customer 4: nine food-store inputs, then a gas-station target. Counting the last ten (target included)
    # would keep it; counting the nine inputs drops it.
    rows = [(4, f"2014-07-{d:02d}", 5.0, 5411) for d in range(1, 10)] + [(4, "2014-07-10", 5.0, 5541)]
    df = pd.DataFrame(rows, columns=["customer_id", "date", "amount", "mcc"])
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = data.map_categories(df["mcc"])
    assert list(data.eligible_customers(df, on_inputs=False)) == [4]
    assert list(data.eligible_customers(df, on_inputs=True)) == []
    assert list(data.eligible_customers(df, on_inputs=True, min_distinct=1)) == [4]


def test_averaging_ties():
    codes = np.array([[2, 2, 1, 1], [3, 3, 3, 0], [0, 0, 1, 1]])
    assert list(baselines.averaging_predictions(codes, "lowest")) == [1, 3, 0]
    assert list(baselines.averaging_predictions(codes)) == [1, 3, 1]       # recency: the tied class seen last


def test_markov_and_last_category():
    m = baselines.Markov1().fit(np.array([[0, 1, 0, 1, 0], [2, 2, 2, 2, 2]]))
    assert list(m.predict(np.array([[3, 0], [3, 1], [3, 2]]))) == [1, 0, 2]
    assert list(baselines.last_category_predictions(np.array([[0, 1], [2, 3]]))) == [1, 3]


def test_ties_on_the_target_date():
    t = data.tie_report(transactions())
    assert t["tied_last_date"] == 0
    df = transactions()
    df.loc[df.index[-1], "date"] = df.loc[df.index[-2], "date"]     # customer 3: last two on the same day
    assert data.tie_report(df)["tied_last_date"] == 1


def test_same_day_transactions_are_ordered_by_time(tmp_path):
    rows = [(7, "2014-07-01", "10:00:00", 1.0, 5411), (7, "2014-07-02", "18:30:00", 2.0, 5541),
            (7, "2014-07-02", "09:15:00", 3.0, 5651), (7, "2014-07-02", None, 4.0, 4812)]
    pd.DataFrame(rows, columns=["customer_id", "date", "time", "amount", "mcc"]).to_csv(tmp_path / "t.csv", index=False)
    tx = data.load_transactions(tmp_path / "t.csv")
    # on 2 July: the transaction without a time first, then by time; the target is the 18:30 purchase
    assert list(tx["amount"]) == [1.0, 4.0, 3.0, 2.0]
    assert tx["category"].iloc[-1] == "Gas stations"

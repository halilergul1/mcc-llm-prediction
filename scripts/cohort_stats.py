"""Descriptive statistics of the two eligible cohorts (Section 3.9, Table 4, Figures 2 and 3, Appendix E).

Reads both banks in the pipeline's input schema (scripts/export_raw.py), applies the cohort filters of
data.eligible_customers, and compares the two eligible cohorts as complete populations: class shares, activity,
the span of the last ten transactions, amounts, demographics, sequence predictability, the population stability
index (PSI) per dimension and the merchant category codes behind each class. Nothing here uses a model.

    python scripts/cohort_stats.py --bank-a raw/bank_a --bank-b raw/bank_b --out results/local/cohort

Outputs (CSV, numbers as numbers): funnel, activity, class_shares, predictability, demographics, psi, mcc_shares;
figures/fig2_activity.pdf and figures/fig3_demographics.pdf.
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from mcc_llm import baselines, config, data

BANKS = ("bank_a", "bank_b")
LEGEND = {"bank_a": "Bank A cohort", "bank_b": "Bank B cohort"}
AGE_BINS, AGE_LABELS = [0, 29, 39, 49, 59, 200], ["<30", "30-39", "40-49", "50-59", "60+"]   # as evaluate_v3.subgroups
LEVELS = {
    "Age band": AGE_LABELS,
    "Gender": ["male", "female"],
    "Marital status": ["married", "single", "divorced or widowed", "unknown"],
    "Education": ["primary school", "middle school", "high school", "university", "graduate school", "unknown"],
    "Occupation": ["private sector employee", "public sector employee", "self-employed", "retired", "not employed",
                   "other occupation"],
    "Income group": ["low", "middle", "high"],
}
GAS = config.CATEGORIES.index("Gas stations")

# figure style (journal widths in inches; Bank A blue, Bank B orange)
COL2 = 6.9
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOUR = {"bank_a": "#2a78d6", "bank_b": "#eb6834"}
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.labelsize": 8,
    "axes.titlesize": 9, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
}


# --------------------------------------------------------------------------- shift measures
def psi(p, q, eps=1e-6) -> float:
    """Population stability index of distribution q (Bank B) against reference p (Bank A)."""
    p, q = np.clip(np.asarray(p, float), eps, None), np.clip(np.asarray(q, float), eps, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def psi_numeric(a, b, n_bins=10) -> float:
    """PSI of a numerical variable, binned on the deciles of the reference (Bank A)."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.concatenate([[-np.inf], edges, [np.inf]])
    return psi(np.histogram(a, bins)[0], np.histogram(b, bins)[0])


# --------------------------------------------------------------------------- one cohort
def averaging_rule(codes: np.ndarray) -> np.ndarray:
    """The Averaging baseline: most frequent input class, ties to the most recent (baselines.averaging_predictions)."""
    return baselines.averaging_predictions(codes)


def cohort(tx_all: pd.DataFrame, demo: pd.DataFrame, months: int) -> dict:
    """Eligible customers, their transactions and per-customer quantities."""
    ids = data.eligible_customers(tx_all, demo)
    tx = tx_all[tx_all["customer_id"].isin(ids)]
    w = data.last_windows(tx, config.TRAIN_LENGTH)
    codes = w.codes()
    inputs, target = codes[:, :-1], codes[:, -1]
    n_txn = tx.groupby("customer_id").size().reindex(w.customer_id).to_numpy()
    span = (w.dates[:, -1] - w.dates[:, 0]) / np.timedelta64(1, "D")
    distinct = np.stack([(inputs == k).any(1) for k in range(config.NUM_CLASSES)], 1).sum(1)
    return {"ids": ids, "tx": tx, "codes": codes, "inputs": inputs, "target": target, "n_txn": n_txn,
            "span": span.astype(float), "distinct": distinct, "months": months,
            "demo": demo.loc[w.customer_id]}


def months_observed(tx: pd.DataFrame) -> int:
    first, last = tx["date"].min().to_period("M"), tx["date"].max().to_period("M")
    return int((last - first).n + 1)


def funnel(tx: pd.DataFrame, demo: pd.DataFrame) -> dict:
    counts = tx.groupby("customer_id").size()
    with_demo = counts[counts.index.isin(demo.index)]
    return {
        "customers with transactions": int(len(counts)),
        "with complete demographics": int(len(with_demo)),
        "with at least 10 transactions": int((with_demo >= config.MIN_TRANSACTIONS).sum()),
        "eligible: at least 2 classes among the 9 inputs": int(len(data.eligible_customers(tx, demo))),
        "earlier rule: at least 2 classes among the last 10 (target included)":
            int(len(data.eligible_customers(tx, demo, on_inputs=False))),
    }


def activity_rows(c: dict, usd_rate: float) -> dict:
    n, span, amount = c["n_txn"], c["span"], c["tx"]["amount"].to_numpy()
    q = lambda v, p: float(np.quantile(v, p))
    return {
        "Customers": len(c["ids"]),
        "Transactions": len(c["tx"]),
        "First date": c["tx"]["date"].min().strftime("%Y-%m-%d"),
        "Last date": c["tx"]["date"].max().strftime("%Y-%m-%d"),
        "Months observed": c["months"],
        "Transactions per customer: median": float(np.median(n)),
        "Transactions per customer: IQR low": q(n, .25), "Transactions per customer: IQR high": q(n, .75),
        "Transactions per customer per month: median": round(float(np.median(n / c["months"])), 1),
        "Customers with at least 15 transactions (%)": round(100 * float((n >= 15).mean()), 1),
        "Customers with at least 20 transactions (%)": round(100 * float((n >= 20).mean()), 1),
        "Span of the last 10 transactions, days: median": float(np.median(span)),
        "Span of the last 10 transactions, days: IQR low": q(span, .25),
        "Span of the last 10 transactions, days: IQR high": q(span, .75),
        "Transaction amount, TL: median": round(float(np.median(amount)), 1),
        "Transaction amount, TL: IQR low": round(q(amount, .25), 1),
        "Transaction amount, TL: IQR high": round(q(amount, .75), 1),
        "TL per USD, period average": usd_rate,
        "Transaction amount, USD: median": round(float(np.median(amount)) / usd_rate, 1),
    }


def predictability_rows(c: dict) -> dict:
    inputs, target = c["inputs"], c["target"]
    last = inputs[:, -1]
    gas_last = last == GAS
    shares = np.bincount(target, minlength=config.NUM_CLASSES) / len(target)
    entropy = float(-(shares[shares > 0] * np.log2(shares[shares > 0])).sum())
    row = {
        "Target equals the last input category (%)": round(100 * float((last == target).mean()), 1),
        "Target equals the most frequent input category (%)": round(100 * float((averaging_rule(inputs) == target).mean()), 1),
        "Inputs with 2 distinct categories (%)": round(100 * float((c["distinct"] == 2).mean()), 1),
        "Inputs with 3 distinct categories (%)": round(100 * float((c["distinct"] == 3).mean()), 1),
        "Inputs with 4 distinct categories (%)": round(100 * float((c["distinct"] == 4).mean()), 1),
        "Distinct categories among the inputs: mean": round(float(c["distinct"].mean()), 2),
        "Category changes within the 10 transactions: mean": round(float((np.diff(c["codes"], axis=1) != 0).sum(1).mean()), 2),
        "Target is Gas stations when the last input is Gas stations (%)": round(100 * float((target[gas_last] == GAS).mean()), 1),
        "Last input is Gas stations (%)": round(100 * float(gas_last.mean()), 1),
        "Target entropy (bits)": round(entropy, 3),
    }
    return row


def reweighted_averaging(a: dict, b: dict) -> float:
    """Accuracy of the Averaging rule on Bank A, with Bank A customers reweighted to Bank B's target class mix."""
    hit = averaging_rule(a["inputs"]) == a["target"]
    share_a = np.bincount(a["target"], minlength=config.NUM_CLASSES) / len(a["target"])
    share_b = np.bincount(b["target"], minlength=config.NUM_CLASSES) / len(b["target"])
    w = (share_b / share_a)[a["target"]]
    return float((hit * w).sum() / w.sum())


def demographic_table(c: dict) -> pd.DataFrame:
    d = c["demo"]
    view = pd.DataFrame({
        "Age band": pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS).astype(str),
        "Gender": d["gender"], "Marital status": d["marital_status"], "Education": d["education"],
        "Occupation": d["occupation"], "Income group": d["income_group"],
    })
    rows = []
    for var, levels in LEVELS.items():
        counts = view[var].value_counts().reindex(levels).fillna(0)
        for level in levels:
            rows.append({"variable": var, "level": level, "share": 100 * counts[level] / counts.sum()})
    return pd.DataFrame(rows)


def mcc_table(c: dict) -> pd.DataFrame:
    t = c["tx"].groupby(["category", "mcc"]).size().rename("transactions").reset_index()
    t["share of all transactions (%)"] = 100 * t["transactions"] / t["transactions"].sum()
    t["share within class (%)"] = 100 * t["transactions"] / t.groupby("category")["transactions"].transform("sum")
    return t.sort_values(["category", "transactions"], ascending=[True, False])


# --------------------------------------------------------------------------- figures
def ecdf_panel(ax, values: dict, title, xlabel, left, fmt="{:,.0f}"):
    """ECDF of one quantity for both cohorts on a log axis, cut to the 0.2nd-99.8th percentile; dots mark medians."""
    probs = np.linspace(0, 1, 2001)
    vals = {k: np.asarray(v, float)[np.isfinite(v) & (np.asarray(v, float) > 0)] for k, v in values.items()}
    lo = min(np.quantile(v, 0.002) for v in vals.values())
    hi = max(np.quantile(v, 0.998) for v in vals.values())
    for k in BANKS:
        v = vals[k]
        ax.plot(np.quantile(v, probs), probs * 100, color=COLOUR[k], lw=1.6, solid_capstyle="round", zorder=3)
        m = float(np.median(v))
        ax.scatter([m], [50], s=34, color=COLOUR[k], edgecolor="white", linewidth=1.2, zorder=4)
        dx, dy, ha, va = (-5, 6, "right", "bottom") if k == left else (5, -6, "left", "top")
        ax.annotate(fmt.format(m), (m, 50), xytext=(dx, dy), textcoords="offset points", color=INK2, fontsize=7.5,
                    ha=ha, va=va, zorder=5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    ax.set_xlim(lo * 0.85, hi * 1.15)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    ax.text(0.02, 0.97, "log scale", transform=ax.transAxes, ha="left", va="top", fontsize=7, color=MUTED)


def figure_activity(cohorts: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), sharey=True)
    ecdf_panel(axes[0], {k: c["n_txn"] for k, c in cohorts.items()}, "Transactions per customer",
               "Transactions in the observation period", left="bank_b")
    ecdf_panel(axes[1], {k: np.clip(c["span"], 1, None) for k, c in cohorts.items()}, "Span of last 10 transactions",
               "Days from first to last transaction", left="bank_a", fmt="{:,.0f} d")
    ecdf_panel(axes[2], {k: c["tx"]["amount"].to_numpy() for k, c in cohorts.items()}, "Transaction amount",
               "Turkish lira (nominal)", left="bank_a", fmt="TL {:,.1f}")
    axes[0].set_ylabel("Cumulative share of customers (%)")
    axes[2].set_ylabel("Cumulative share of transactions (%)")
    axes[2].yaxis.set_label_position("right")
    fig.legend(handles=[Line2D([], [], color=COLOUR[k], lw=1.6, label=LEGEND[k]) for k in BANKS], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.09), columnspacing=1.6)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(path)
    plt.close(fig)


def figure_demographics(demo: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.1))
    for ax, (var, levels) in zip(axes.ravel(), LEVELS.items()):
        t = demo[demo["variable"] == var].set_index("level")
        lv = [lev for lev in levels if t.loc[lev, ["bank_a", "bank_b"]].max() >= 0.5]       # hide empty 'unknown' rows
        a, b = t.loc[lv, "bank_a"].to_numpy(), t.loc[lv, "bank_b"].to_numpy()
        y = np.arange(len(lv))[::-1]
        sc = len(lv) / 6
        h, g = 0.30 * sc, 0.035 * sc
        ax.barh(y + h / 2 + g, a, height=h, color=COLOUR["bank_a"], zorder=3)
        ax.barh(y - h / 2 - g, b, height=h, color=COLOUR["bank_b"], zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([lev[0].upper() + lev[1:] for lev in lv])
        ax.set_ylim(-0.7, len(lv) - 0.3)
        ax.set_xlim(0, max(a.max(), b.max()) * 1.12)
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
        ax.set_title(var)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, integer=True))
    for ax in axes[1]:
        ax.set_xlabel("Share of customers (%)")
    fig.legend(handles=[mpl.patches.Patch(color=COLOUR[k], label=LEGEND[k]) for k in BANKS], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.035), columnspacing=1.6)
    fig.tight_layout(w_pad=1.6, h_pad=1.6)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank-a", type=Path, default=Path("raw/bank_a"), help="folder with transactions.csv and demographics.csv")
    p.add_argument("--bank-b", type=Path, default=Path("raw/bank_b"))
    p.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"), help="checks the eligible counts")
    p.add_argument("--usd-a", type=float, default=1.8401, help="TL per USD, period average of Bank A (from ECB euro reference rates)")
    p.add_argument("--usd-b", type=float, default=2.3868, help="TL per USD, period average of Bank B")
    p.add_argument("--season", nargs=2, default=["2015-04-01", "2015-06-30"],
                   help="Bank B window of the same season as Bank A's observation period")
    p.add_argument("--out", type=Path, default=Path("results/local/cohort"))
    args = p.parse_args()
    mpl.rcParams.update(STYLE)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    raw = {}
    for bank, folder in zip(BANKS, (args.bank_a, args.bank_b)):
        tx = data.load_transactions(folder / "transactions.csv")
        demo = data.load_demographics(folder / "demographics.csv")
        raw[bank] = (tx, demo)
        print(f"{bank}: {len(tx):,} transactions, {tx['customer_id'].nunique():,} customers", flush=True)

    fun = pd.DataFrame({b: funnel(*raw[b]) for b in BANKS})
    fun.index.name = "step"
    fun.to_csv(args.out / "funnel.csv")
    print(fun.to_string())

    cohorts = {b: cohort(raw[b][0], raw[b][1], months_observed(raw[b][0])) for b in BANKS}
    if args.splits.exists():
        expected = json.loads(args.splits.read_text())["eligible"]
        assert len(cohorts["bank_a"]["ids"]) == expected["training bank"], "Bank A cohort differs from splits.json"
        assert len(cohorts["bank_b"]["ids"]) == expected["test bank"], "Bank B cohort differs from splits.json"

    rates = {"bank_a": args.usd_a, "bank_b": args.usd_b}
    act = pd.DataFrame({b: activity_rows(cohorts[b], rates[b]) for b in BANKS})
    # the same season at Bank B: transactions of the Bank A months only, same filters
    tx_b, demo_b = raw["bank_b"]
    season = tx_b[(tx_b["date"] >= args.season[0]) & (tx_b["date"] <= args.season[1])]
    c_season = cohort(season, demo_b, months_observed(season))
    act.loc["Same-season window: customers", :] = [np.nan, len(c_season["ids"])]
    act.loc["Same-season window: span of the last 10 transactions, days: median", :] = [
        float(np.median(cohorts["bank_a"]["span"])), float(np.median(c_season["span"]))]
    act.index.name = "statistic"
    act.to_csv(args.out / "activity.csv")
    print(act.to_string())

    shares = []
    for b in BANKS:
        c = cohorts[b]
        all_tx = c["tx"]["category"].value_counts(normalize=True)
        tgt = np.bincount(c["target"], minlength=config.NUM_CLASSES) / len(c["target"])
        shares.append({"view": "All transactions", "bank": b, **{k: round(100 * float(all_tx.get(k, 0)), 1) for k in config.CATEGORIES}})
        shares.append({"view": "Prediction targets", "bank": b, **{k: round(100 * float(tgt[i]), 1) for i, k in enumerate(config.CATEGORIES)}})
    shares = pd.DataFrame(shares)
    shares.to_csv(args.out / "class_shares.csv", index=False)
    print(shares.to_string(index=False))

    pred = pd.DataFrame({b: predictability_rows(cohorts[b]) for b in BANKS})
    acc_a = float((averaging_rule(cohorts["bank_a"]["inputs"]) == cohorts["bank_a"]["target"]).mean())
    pred.loc["Averaging rule on Bank A reweighted to Bank B's target mix (%)", :] = [
        round(100 * reweighted_averaging(cohorts["bank_a"], cohorts["bank_b"]), 1), np.nan]
    pred.index.name = "statistic"
    pred.to_csv(args.out / "predictability.csv")
    print(pred.to_string())
    print(f"(Averaging rule on Bank A, unweighted: {100 * acc_a:.1f}%)")

    demo_tab = demographic_table(cohorts["bank_a"]).merge(demographic_table(cohorts["bank_b"]), on=["variable", "level"],
                                                           suffixes=("_a", "_b"))
    demo_tab = demo_tab.rename(columns={"share_a": "bank_a", "share_b": "bank_b"})
    age = pd.DataFrame([{"variable": "Age", "level": "mean", "bank_a": cohorts["bank_a"]["demo"]["age"].mean(),
                         "bank_b": cohorts["bank_b"]["demo"]["age"].mean()}])
    pd.concat([demo_tab, age]).round(1).to_csv(args.out / "demographics.csv", index=False)

    a, b = cohorts["bank_a"], cohorts["bank_b"]
    rows = {
        "Class shares, all transactions": psi(a["tx"]["category"].value_counts().reindex(config.CATEGORIES),
                                              b["tx"]["category"].value_counts().reindex(config.CATEGORIES)),
        "Class shares, prediction targets": psi(np.bincount(a["target"], minlength=4), np.bincount(b["target"], minlength=4)),
        "Transactions per customer": psi_numeric(a["n_txn"], b["n_txn"]),
        "Span of the last 10 transactions": psi_numeric(a["span"], b["span"]),
        "Span of the last 10 transactions, same-season Bank B window": psi_numeric(a["span"], c_season["span"]),
        "Transaction amount": psi_numeric(a["tx"]["amount"].to_numpy(), b["tx"]["amount"].to_numpy()),
    }
    for var in LEVELS:
        t = demo_tab[demo_tab["variable"] == var]
        rows[var] = psi(t["bank_a"].to_numpy(), t["bank_b"].to_numpy())
    psi_tab = pd.Series(rows, name="PSI").round(3).sort_values(ascending=False)
    psi_tab.index.name = "dimension"
    psi_tab.to_csv(args.out / "psi.csv")
    print(psi_tab.to_string())

    mcc = pd.concat([mcc_table(cohorts[bk]).assign(bank=bk) for bk in BANKS])
    mcc.round(2).to_csv(args.out / "mcc_shares.csv", index=False)

    figure_activity(cohorts, args.out / "figures" / "fig2_activity.pdf")
    figure_demographics(demo_tab, args.out / "figures" / "fig3_demographics.pdf")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

"""Convert a bank's raw files into the input schema of the pipeline (the step before prepare_data.py).

    python scripts/export_raw.py --spec bank_a.json --out raw/bank_a
    python scripts/export_raw.py --spec bank_b.json --out raw/bank_b

Writes <out>/transactions.csv (customer_id, date, time, amount, mcc) and <out>/demographics.csv
(customer_id, age, gender, marital_status, education, occupation, income). The spec is a JSON file
that says where the raw files are, which raw columns hold what, and how the demographic labels are
worded; examples/export_spec.example.json documents it. The bank data and the real specs are never
part of this repository.

Rules, the same for every bank:
  * every raw column is read as text; a row whose customer identifier is not an integer is a broken
    line and is dropped;
  * a row is dropped if one of the `require` columns is empty (for example a transaction without a
    merchant category description), or if its date, amount or merchant category code is missing;
  * dates become YYYY-MM-DD and times HH:MM:SS; a transaction without a recorded time keeps an empty
    time (the pipeline then puts it first on its day);
  * amounts keep their currency and are not rescaled;
  * each demographic label is replaced by the word given in the spec, so that two banks share one
    vocabulary; a label that the spec does not list stops the export (nothing is dropped silently),
    unless the field has a "*" entry, which is then used for every unlisted label;
  * a customer with a missing demographic value is kept in the file with the value empty; the
    pipeline's "complete demographics" filter removes these customers later.
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

TEXT_FIELDS = ("gender", "marital_status", "education", "occupation")


def read_raw(section: dict, columns: list[str]) -> pd.DataFrame:
    """All files of one section, in the order given (a glob pattern expands in sorted order)."""
    files = [f for pattern in section["files"] for f in (sorted(glob.glob(pattern)) or [pattern])]
    options = {"dtype": str, **section.get("read_csv", {})}        # pandas reads "", "NA", "NULL", ... as missing
    parts = [pd.read_csv(f, usecols=lambda c: c in columns, **options) for f in files]
    df = pd.concat(parts, ignore_index=True)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"raw columns {missing} not found in {files[0]}")
    print(f"  {len(files)} file(s), {len(df):,} rows read", flush=True)
    return df


def is_empty(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def number(s: pd.Series, decimal: str) -> pd.Series:
    """Numbers written with `decimal` as the decimal separator (no thousands separators)."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(decimal, ".", regex=False), errors="coerce")


def parsed(s: pd.Series, fmt: str, out: str) -> pd.Series:
    """Parse the distinct values once (dates and times repeat a lot); unparsable or empty values become missing."""
    values = pd.Series(s.unique())
    text = pd.to_datetime(values.astype(str).str.strip(), format=fmt, errors="coerce").dt.strftime(out)
    return s.map(dict(zip(values, text)))


def export_transactions(section: dict, out: Path) -> dict:
    col = section["columns"]
    require = section.get("require", [])
    df = read_raw(section, list(col.values()) + require)
    report = {"rows read": len(df)}

    ok = df[col["customer_id"]].astype(str).str.fullmatch(r"\s*\d+\s*")
    report["dropped: customer identifier is not an integer (broken line)"] = int((~ok).sum())
    df = df[ok]
    for r in require:
        ok = ~is_empty(df[r])
        report[f"dropped: empty {r}"] = int((~ok).sum())
        df = df[ok]

    tx = pd.DataFrame({
        "customer_id": df[col["customer_id"]].astype("int64"),
        "date": parsed(df[col["date"]], section["date_format"], "%Y-%m-%d"),
        "time": parsed(df[col["time"]], section["time_format"], "%H:%M:%S") if "time" in col else None,
        "amount": number(df[col["amount"]], section.get("decimal", ".")),
        "mcc": pd.to_numeric(df[col["mcc"]], errors="coerce"),
    })
    if "time" not in col:
        tx = tx.drop(columns="time")
    else:
        report["transactions without a recorded time (kept)"] = int(tx["time"].isna().sum())
    ok = tx[["date", "amount", "mcc"]].notna().all(axis=1)
    report["dropped: missing date, amount or merchant category code"] = int((~ok).sum())
    tx = tx[ok]
    tx["mcc"] = tx["mcc"].astype(int)
    report["rows written"], report["customers"] = len(tx), int(tx["customer_id"].nunique())
    report["first date"], report["last date"] = tx["date"].min(), tx["date"].max()

    out.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out / "transactions.csv", index=False)
    for k, v in report.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    return report


def export_demographics(section: dict, out: Path) -> dict:
    col = section["columns"]
    df = read_raw(section, list(col.values()))
    ok = df[col["customer_id"]].astype(str).str.fullmatch(r"\s*\d+\s*")
    df = df[ok]
    demo = pd.DataFrame({"customer_id": df[col["customer_id"]].astype("int64"),
                         "age": pd.to_numeric(df[col["age"]], errors="coerce")})
    counts = {}
    for field in TEXT_FIELDS:
        raw = df[col[field]].astype(str).str.strip().where(~is_empty(df[col[field]]))
        words = section["labels"][field]
        unlisted = sorted(set(raw.dropna()) - set(words))
        if unlisted and "*" not in words:
            raise SystemExit(f"{field}: the spec lists no word for the raw labels {unlisted}")
        demo[field] = raw.map(lambda v: words.get(v, words.get("*")), na_action="ignore")
        counts[field] = demo[field].value_counts(dropna=False).to_dict()
    demo["income"] = number(df[col["income"]], section.get("decimal", "."))
    demo["age"] = demo["age"].astype("Int64")
    complete = int(demo.notna().all(axis=1).sum())

    out.mkdir(parents=True, exist_ok=True)
    demo.to_csv(out / "demographics.csv", index=False)
    print(f"  customers written: {len(demo):,}; with complete demographics: {complete:,}")
    for field, c in counts.items():
        print(f"  {field}: " + ", ".join(f"{k} {v:,}" for k, v in c.items()))
    return {"customers": len(demo), "complete": complete, "levels": {f: {str(k): v for k, v in c.items()} for f, c in counts.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", type=Path, required=True, help="JSON description of the bank's raw files")
    p.add_argument("--out", type=Path, required=True, help="output folder, e.g. raw/bank_a")
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = {}
    print("transactions:")
    report["transactions"] = export_transactions(spec["transactions"], args.out)
    if "demographics" in spec:
        print("demographics:")
        report["demographics"] = export_demographics(spec["demographics"], args.out)
    (args.out / "export_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()

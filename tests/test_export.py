"""scripts/export_raw.py on a small made-up bank export."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mcc_llm import data

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_raw.py"


def test_export_raw(tmp_path):
    (tmp_path / "tx.txt").write_text(
        "CUST\tDAY\tCLOCK\tAMT\tCODE\tTEXT\n"
        "1\t4/3/2013\t12.41.37\t40\t5813\tBars\n"
        "1\t4/3/2013\t \t21,55\t5411\tGrocery stores\n"                # no recorded time: kept, first on its day
        "continuation of a broken line\tx\n"                            # customer identifier is not an integer
        "2\t4/6/2013\t09.05.00\t7,5\t5541\t\n"                         # required text missing: dropped
        "2\t4/7/2013\t23.59.59\t100\t5651\tFamily clothing\n", encoding="utf-8")
    (tmp_path / "customers.txt").write_text(
        "CUST\tAGE\tSEX\tMARITAL\tSCHOOL\tJOB\tINCOME\n"
        "1\t49\tM\tMarried\tCollege\tPublic Servant\t3762,54\n"
        "2\t31\tF\tDivorced\tHigh School\tStudent\t\n", encoding="utf-8")
    spec = {
        "transactions": {
            "files": [str(tmp_path / "tx.txt")], "read_csv": {"sep": "\t", "on_bad_lines": "skip"},
            "columns": {"customer_id": "CUST", "date": "DAY", "time": "CLOCK", "amount": "AMT", "mcc": "CODE"},
            "date_format": "%m/%d/%Y", "time_format": "%H.%M.%S", "decimal": ",", "require": ["TEXT"]},
        "demographics": {
            "files": [str(tmp_path / "customers.txt")], "read_csv": {"sep": "\t"},
            "columns": {"customer_id": "CUST", "age": "AGE", "gender": "SEX", "marital_status": "MARITAL",
                        "education": "SCHOOL", "occupation": "JOB", "income": "INCOME"},
            "decimal": ",",
            "labels": {"gender": {"M": "male", "F": "female"},
                       "marital_status": {"Married": "married", "Divorced": "divorced or widowed"},
                       "education": {"College": "university", "High School": "high school"},
                       "occupation": {"Public Servant": "public sector employee", "*": "other occupation"}}},
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / "raw"
    subprocess.run([sys.executable, str(SCRIPT), "--spec", str(tmp_path / "spec.json"), "--out", str(out)], check=True)

    tx = pd.read_csv(out / "transactions.csv")
    assert list(tx.columns) == ["customer_id", "date", "time", "amount", "mcc"]
    assert list(tx["customer_id"]) == [1, 1, 2] and list(tx["mcc"]) == [5813, 5411, 5651]
    assert list(tx["date"]) == ["2013-04-03", "2013-04-03", "2013-04-07"] and list(tx["amount"]) == [40.0, 21.55, 100.0]
    assert tx["time"].tolist()[0] == "12:41:37" and pd.isna(tx["time"].tolist()[1])

    ordered = data.load_transactions(out / "transactions.csv")           # the untimed purchase comes first on its day
    assert list(ordered["mcc"]) == [5411, 5813, 5651]

    demo = pd.read_csv(out / "demographics.csv")
    assert list(demo["gender"]) == ["male", "female"] and list(demo["occupation"]) == ["public sector employee", "other occupation"]
    assert list(demo["education"]) == ["university", "high school"] and demo["income"].tolist()[0] == 3762.54
    assert list(demo.dropna()["customer_id"]) == [1]                    # customer 2 has no income: incomplete record


def test_unlisted_label_stops_the_export(tmp_path):
    (tmp_path / "tx.csv").write_text("id,day,amount,code\n1,2014-07-01,5.0,5411\n")
    (tmp_path / "c.csv").write_text("id,age,sex,marital,school,job,income\n1,40,X,Married,College,Clerk,10\n")
    spec = {"transactions": {"files": [str(tmp_path / "tx.csv")], "date_format": "%Y-%m-%d",
                             "columns": {"customer_id": "id", "date": "day", "amount": "amount", "mcc": "code"}},
            "demographics": {"files": [str(tmp_path / "c.csv")],
                             "columns": {"customer_id": "id", "age": "age", "gender": "sex", "marital_status": "marital",
                                         "education": "school", "occupation": "job", "income": "income"},
                             "labels": {"gender": {"M": "male", "F": "female"}, "marital_status": {"Married": "married"},
                                        "education": {"College": "university"}, "occupation": {"*": "other occupation"}}}}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    run = subprocess.run([sys.executable, str(SCRIPT), "--spec", str(tmp_path / "spec.json"), "--out", str(tmp_path / "raw")],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "gender" in run.stderr
    assert list(pd.read_csv(tmp_path / "raw" / "transactions.csv").columns) == ["customer_id", "date", "amount", "mcc"]

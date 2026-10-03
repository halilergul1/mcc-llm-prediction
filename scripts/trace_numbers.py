"""Find numbers in the manuscript or the letters that no results file contains.

Every decimal in the given text files (LaTeX, Markdown or plain text) is looked up among the values of
every CSV under the results folders, rounded to the same number of decimals. Signed differences are looked
up by absolute value. What is left needs a human: a hand-typed number, a stale one, or a derived one
(then add the derivation to the table script).

    python scripts/trace_numbers.py manuscript/main.tex letters/response.tex        # searches the metric tables only
    python scripts/trace_numbers.py main.tex --ignore 0.05 0.95 1.96   # thresholds, not results
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NUMBER = re.compile(r"(?<![\w.])[-+−]?\d*\.\d+(?![\w.])")


def known_values(folders):
    values = {1: set(), 2: set(), 3: set(), 4: set()}
    for folder in folders:
        for path in Path(folder).rglob("*.csv"):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            nums = df.select_dtypes("number").to_numpy(float).ravel()
            nums = np.abs(nums[np.isfinite(nums)])
            for d in values:
                values[d].update(np.round(nums, d).tolist())
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texts", type=Path, nargs="+")
    p.add_argument("--results", type=Path, nargs="+", default=[Path("results/metrics"), Path("results/metrics_v3")],
                   help="metric tables only: prediction files hold thousands of probabilities and would match anything")
    p.add_argument("--ignore", nargs="*", default=["0.05", "0.95", "0.80", "1.96", "2.80", "0.10", "0.25"])
    args = p.parse_args()
    values = known_values(args.results)
    missing = 0
    for text in args.texts:
        for lineno, line in enumerate(text.read_text(encoding="utf-8").splitlines(), 1):
            for m in NUMBER.finditer(line):
                raw = m.group().replace("−", "-").lstrip("+-")
                if raw in args.ignore:
                    continue
                decimals = len(raw.split(".")[1])
                if decimals not in values:
                    continue
                if round(float(raw), decimals) not in values[decimals]:
                    missing += 1
                    print(f"{text}:{lineno}: {m.group()}   ...{line.strip()[:90]}")
    print(f"\n{missing} numbers not found in any results file")


if __name__ == "__main__":
    main()

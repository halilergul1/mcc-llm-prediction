"""SHA-256 of every file under the given folders (the data splits, the trained models, the prediction files).

    python scripts/manifest.py runs --out results/MANIFEST_models
    python scripts/manifest.py data/processed --check results/MANIFEST_data          # exit 1 on any mismatch

Commit the manifest of the trained models before the confirmation scoring (the freeze of ANALYSIS_PLAN.md),
and the manifest of the prediction files right after it.
"""
import argparse
import hashlib
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", type=Path, nargs="+")
    p.add_argument("--out", type=Path)
    p.add_argument("--check", type=Path)
    args = p.parse_args()
    lines = [f"{digest(f)}  {f.as_posix()}" for folder in args.folders for f in sorted(folder.rglob("*")) if f.is_file()]
    if args.check:
        recorded = set(args.check.read_text().splitlines())
        now = set(lines)
        missing, changed = recorded - now, now - recorded
        for line in sorted(missing):
            print("recorded but not found or changed:", line)
        for line in sorted(changed):
            print("new or changed:", line)
        if not missing:
            print(f"{len(recorded)} recorded files match")
        sys.exit(1 if missing else 0)
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"{len(lines)} files -> {args.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

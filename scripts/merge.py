"""Merge a run's LoRA adapter into the bf16 base weights (<run>/merged), for scoring.

Train with --no-merge to save disk (a merged Qwen3.5-9B is about 18 GB; an adapter is far smaller),
merge just before scoring, and delete <run>/merged afterwards; the adapter stays and is what the
freeze manifest hashes.

    python scripts/merge.py --family qwen --run runs/qwen-full
    python scripts/merge.py --family mistral --run runs/mistral-full --base mistralai/Mistral-7B-Instruct-v0.3
"""
import argparse
from pathlib import Path

from mcc_llm import config, llm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=llm.FAMILIES)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--base", help="base model id or path (default: config.BASE_MODELS[family])")
    args = p.parse_args()
    base = args.base or config.BASE_MODELS[args.family]
    out = llm.merge_adapter(base, args.run / "adapter", args.run / "merged", args.family)
    print(f"merged: {out}")


if __name__ == "__main__":
    main()

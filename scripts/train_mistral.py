"""LoRA fine-tuning of Mistral-7B-Instruct-v0.3 on the training bank (Sections 3.3 to 3.5, Table 2).

One instruction sample per training customer: the last 9 transactions form the Task Input
and the 10th, most recent one is the Instruction Output. The base model is loaded in 8 bits
and frozen; the chat has a user and an assistant turn. After training, the adapter is merged
into the bf16 base weights (<run-dir>/merged) for scoring with scripts/predict_llm.py; with
--no-merge only the adapter is kept and scripts/merge.py merges it later. A run that stopped
resumes from its last checkpoint when the same command is started again.

Examples:
    python scripts/train_mistral.py                                # full instruction
    python scripts/train_mistral.py --seed 2                       # training-seed study (seeds 42, 2, 3)
"""
import argparse
from pathlib import Path

from mcc_llm import config, data, llm, prompts

FAMILY = "mistral"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--model-id", default=config.BASE_MODELS[FAMILY])
    p.add_argument("--condition", default="full", choices=list(prompts.CONDITIONS))
    p.add_argument("--drop-field", choices=prompts.DEMOGRAPHIC_FIELDS, help="demographic field removed from the prompt")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--run-dir", type=Path, help="default: runs/<family>-<condition>[-minus-<field>][-seed<seed>]")
    p.add_argument("--no-8bit", action="store_true", help="load the base model in bf16 instead of 8 bits")
    p.add_argument("--no-merge", action="store_true", help="keep only the LoRA adapter")
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many steps (quick checks only)")
    p.add_argument("--variable-lengths", type=int, nargs=2, metavar=("MIN", "MAX"),
                   help="one window per customer with a random input length in [MIN, MAX]")
    p.add_argument("--max-length", type=int, help=f"maximum sample length in tokens (default {config.SFT['max_length']})")
    args = p.parse_args()

    run_dir = args.run_dir or Path("runs") / llm.run_name(FAMILY, args.condition, args.drop_field, args.seed)
    train_tx, train_demo = data.load_split(args.data, "train")
    ctx = prompts.get_context(args.condition, args.drop_field)
    if args.variable_lengths:
        samples = [s for w in data.variable_windows(train_tx, *args.variable_lengths, seed=args.seed)
                   for s in prompts.build_samples(w, train_demo, ctx)]
        run_dir = args.run_dir or Path(f"{run_dir}-varlen")
    else:
        samples = prompts.build_samples(data.last_windows(train_tx, config.TRAIN_LENGTH), train_demo, ctx)
    print(f"{len(samples):,} training samples; first Task Input:\n{samples[0]['input']}\n", flush=True)

    adapter = llm.finetune(args.model_id, FAMILY, samples, run_dir, seed=args.seed,
                           load_in_8bit=not args.no_8bit, max_steps=args.max_steps, max_length=args.max_length)
    print(f"adapter: {adapter}")
    if not args.no_merge:
        print(f"merged model: {llm.merge_adapter(args.model_id, adapter, run_dir / 'merged', FAMILY)}")


if __name__ == "__main__":
    main()

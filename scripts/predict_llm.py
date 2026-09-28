"""Predict the test customers with a fine-tuned or raw LLM at every input length (Sections 3.4 and 3.8).

The prompt is built with the same template as in training, at input length X, for every
test customer with at least X + 1 transactions. Greedy decoding, at most 10 new tokens,
served with vLLM (use --backend transformers where vLLM is not available). The raw output is
kept in the `prediction` column and mapped to one of the four classes in `parsed_prediction`;
unreadable outputs are counted and mapped to Other.

Examples:
    python scripts/predict_llm.py --family qwen --model runs/qwen-full/merged
    python scripts/predict_llm.py --family qwen --model runs/qwen-seq/merged --condition seq
    python scripts/predict_llm.py --family qwen --model Qwen/Qwen3.5-9B --name qwen-raw --lengths 9
    python scripts/predict_llm.py --family qwen --model runs/qwen-full/merged --no-customer-id --lengths 9
    python scripts/predict_llm.py --family mistral --model runs/mistral-full/merged
"""
import argparse
from pathlib import Path

from mcc_llm import config, data, llm, predictions, prompts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=llm.FAMILIES)
    p.add_argument("--model", required=True, help="merged model directory, or a base model id for the raw model")
    p.add_argument("--condition", default="full", choices=list(prompts.CONDITIONS))
    p.add_argument("--drop-field", choices=prompts.DEMOGRAPHIC_FIELDS)
    p.add_argument("--no-customer-id", action="store_true", help="remove the 'I am <id>.' sentence (no retraining)")
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--name", help="prediction file stem; default derived from family and condition")
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/predictions"))
    p.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    args = p.parse_args()

    name = args.name or llm.run_name(args.family, args.condition, args.drop_field, customer_id=not args.no_customer_id)
    ctx = prompts.get_context(args.condition, args.drop_field, customer_id=False if args.no_customer_id else None)
    tokenizer = llm.load_tokenizer(args.model, args.family)
    test_tx, test_demo = data.load_split(args.data, "test")

    batches = {}
    for x in args.lengths:
        samples = prompts.build_samples(data.last_windows(test_tx, x), test_demo, ctx)
        batches[x] = (samples, [llm.render_prompt(tokenizer, s, args.family) for s in samples])
    print(f"example prompt:\n{batches[args.lengths[0]][1][0]}\n", flush=True)

    all_prompts = [prompt for _, (_, prompt_list) in batches.items() for prompt in prompt_list]
    outputs = iter(llm.generate(args.model, all_prompts, backend=args.backend, family=args.family))
    for x, (samples, _) in batches.items():
        raw = [" ".join(next(outputs).split()) for _ in samples]
        parsed = [llm.parse_category(r) for r in raw]
        unreadable = sum(v is None for v in parsed)
        path = predictions.write(args.out / predictions.file_name(name, x), [s["customer_id"] for s in samples],
                                 [s["response"].rstrip(".") for s in samples], [v or "Other" for v in parsed], raw)
        print(f"length {x} (N = {len(samples):,}): {path}; unreadable outputs mapped to Other: {unreadable}")


if __name__ == "__main__":
    main()

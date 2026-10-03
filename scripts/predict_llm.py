"""Predict the evaluation customers with a fine-tuned or raw LLM (Sections 3.4, 3.5 and 3.8).

The prediction is the most probable of the four answers under the model, and the four class
probabilities are written to the prediction file. Two scorers:
    --score first     first-token scoring, one forward pass per prompt (fast; for fine-tuned models)
    --score sequence  exact scoring of each full answer string (4 passes per prompt; for raw models)
    --check 200       before scoring, compare the two scorers on 200 prompts and print the agreement
    --generate        free greedy generation INSTEAD of scoring (a separate run with vLLM: the scorer and
                      vLLM do not fit on one 32 GB card together); writes <name>-gen files, to count
                      answers that are not a category. Run it on bank_b_dev at length 9.
Raw models: pass the base model id and --name, and add --few-shot for the 16-shot variant.

By default the splits that may be scored before the freeze are scored (config.PRE_SPLITS). The
confirmation splits are scored once, after the freeze: --splits bank_b_confirm bank_b_unfiltered
(scripts/score_all.sh confirm does this for every model).

Examples:
    python scripts/predict_llm.py --family qwen --model runs/qwen-full/merged --check 200
    python scripts/predict_llm.py --family qwen --model runs/qwen-seq/merged --condition seq
    python scripts/predict_llm.py --family qwen --model runs/qwen-neutral/merged --condition neutral
    python scripts/predict_llm.py --family qwen --model Qwen/Qwen3.5-9B --name qwen-raw --score sequence --lengths 9
    python scripts/predict_llm.py --family qwen --model Qwen/Qwen3.5-9B --name qwen-raw16 --score sequence --few-shot --lengths 9
"""
import argparse
import json
import time
from pathlib import Path


from mcc_llm import config, data, llm, predictions, prompts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=llm.FAMILIES)
    p.add_argument("--model", required=True, help="merged model directory, or a base model id for the raw model")
    p.add_argument("--condition", default="full", choices=list(prompts.CONDITIONS))
    p.add_argument("--drop-field", choices=prompts.DEMOGRAPHIC_FIELDS)
    p.add_argument("--splits", nargs="+", default=list(config.PRE_SPLITS))
    p.add_argument("--lengths", type=int, nargs="+", default=list(config.EVAL_LENGTHS))
    p.add_argument("--name", help="prediction file stem; default derived from family and condition")
    p.add_argument("--score", choices=["first", "sequence"], default="first")
    p.add_argument("--check", type=int, default=0, help="compare the two scorers on this many prompts first")
    p.add_argument("--few-shot", action="store_true", help="prepend 16 fixed training examples (raw models)")
    p.add_argument("--generate", action="store_true", help="free generation instead of scoring (<name>-gen)")
    p.add_argument("--backend", default="vllm", choices=["vllm", "transformers"], help="engine of --generate")
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/predictions"))
    args = p.parse_args()

    name = args.name or llm.run_name(args.family, args.condition, args.drop_field)
    ctx = prompts.get_context(args.condition, args.drop_field)
    labels = prompts.label_words(ctx)
    if args.generate:
        tokenizer, lm = llm.load_tokenizer(args.model, args.family), None
    else:
        tokenizer, lm = llm.load_scorer(args.model, args.family)
    examples = None
    if args.few_shot:
        train_tx, train_demo = data.load_split(args.data, "train")
        train = prompts.build_samples(data.last_windows(train_tx, config.TRAIN_LENGTH), train_demo, ctx)
        examples = prompts.few_shot_examples(train)

    scorer = llm.first_token_scores if args.score == "first" else llm.sequence_scores
    log = {"model": str(args.model), "name": name, "condition": args.condition, "score": args.score,
           "few_shot": args.few_shot, "labels": labels, "runs": []}
    for split in args.splits:
        tx, demo = data.load_split(args.data, split)
        for x in args.lengths:
            w = data.last_windows(tx, x)
            if len(w) == 0:
                continue
            samples = prompts.build_samples(w, demo, ctx)
            if examples:
                samples = [prompts.with_few_shot(s, examples) for s in samples]
            texts = [llm.render_prompt(tokenizer, s, args.family) for s in samples]
            if args.generate:
                answers = [llm.answer(t) for t in llm.generate(args.model, texts, backend=args.backend,
                                                               family=args.family)]
                names = [config.CATEGORIES[labels.index(a)] if a in labels else a for a in answers]
                predictions.write(args.out / split / predictions.file_name(f"{name}-gen", x), w.customer_id,
                                  w.target, names)
                bad = sum(a not in labels for a in answers)
                log["runs"].append({"split": split, "length": x, "N": len(w), "not_a_category": bad})
                print(f"{split}, length {x}: free generation, {bad} answers are not a category", flush=True)
                continue
            if args.check and not log.get("check"):
                log["check"] = llm.check_scorers(tokenizer, lm, texts[:args.check], labels)
                print(f"scorer check: {log['check']}", flush=True)
            if args.check and log.get("check", {}).get("first_token_usable") is False and scorer is llm.first_token_scores:
                print("first-token scoring unusable here; falling back to exact sequence scoring", flush=True)
                scorer, log["score"] = llm.sequence_scores, "sequence (fallback)"
            start = time.perf_counter()
            try:
                proba = llm.to_probabilities(scorer(tokenizer, lm, texts, labels))
            except ValueError as err:                     # two answers share their first token
                if scorer is not llm.first_token_scores:
                    raise
                print(f"{err}; falling back to exact sequence scoring", flush=True)
                scorer, log["score"] = llm.sequence_scores, "sequence (fallback)"
                proba = llm.to_probabilities(scorer(tokenizer, lm, texts, labels))
            seconds = time.perf_counter() - start
            path = predictions.write(args.out / split / predictions.file_name(name, x), w.customer_id, w.target,
                                     predictions.to_names(proba.argmax(1)), probs=proba)
            log["runs"].append({"split": split, "length": x, "N": len(w), "seconds": round(seconds, 1)})
            print(f"{split}, length {x} (N = {len(w):,}, {seconds / max(len(w), 1) * 1000:.0f} ms each): {path}", flush=True)
    log_path = args.out / f"{name}{'-gen' if args.generate else ''}.scoring.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    earlier = json.loads(log_path.read_text()) if log_path.exists() else []      # one entry per invocation
    log_path.write_text(json.dumps(earlier + [log], indent=1, default=str))


if __name__ == "__main__":
    main()

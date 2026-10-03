"""LoRA fine-tuning (Section 3.4, Table 2) and prediction for Qwen3.5-9B and Mistral-7B-Instruct.

Chat format: a user turn with the Instruction Input (Task Instruction and Task Input) and
an assistant turn with the Instruction Output, through each model's chat template; for Qwen
the template's thinking mode is disabled. The loss is the log-likelihood of the Instruction
Output tokens (Equation 1).

Training uses the frozen base model in bf16, without quantisation, with PEFT and TRL; an 8-bit copy
(bitsandbytes) is an option. For prediction the adapter is merged into the same bf16 base weights and
the model scores the four answers (label scoring, below): the prediction is the most probable answer,
and the four probabilities are kept. Free greedy generation (at most 10 new tokens) is a check only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config

FAMILIES = ("qwen", "mistral")


def run_name(family: str, condition: str, drop_field: str | None = None, seed: int = config.SEED) -> str:
    """Default name of a fine-tuning run and of its prediction files, e.g. qwen-seq-seed2."""
    name = f"{family}-{condition}"
    if drop_field:
        name += f"-minus-{drop_field.replace('_', '-')}"
    return name if seed == config.SEED else f"{name}-seed{seed}"


# --------------------------------------------------------------------------- prompts
def load_tokenizer(model: str | Path, family: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model))
    if family == "qwen":
        tokenizer.eos_token = "<|im_end|>"
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = "<|endoftext|>"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chat_messages(sample: dict) -> list[dict]:
    return [{"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"}]


def render_prompt(tokenizer, sample: dict, family: str) -> str:
    """Chat-formatted prompt ending where the answer starts. Used for training and inference alike."""
    extra = {"enable_thinking": False} if family == "qwen" else {}
    text = tokenizer.apply_chat_template(chat_messages(sample), tokenize=False,
                                         add_generation_prompt=True, **extra)
    if family == "qwen" and "<think>" in (tokenizer.chat_template or "") and not text.endswith("</think>\n\n"):
        text += "<think>\n\n</think>\n\n"          # thinking disabled: empty reasoning block
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]     # the tokenizer adds BOS itself
    return text


def answer(text: str) -> str:
    """The generated answer without surrounding space and its final period, e.g. 'Clothing.' -> 'Clothing'."""
    return text.strip().removesuffix(".")


# --------------------------------------------------------------------------- fine-tuning
def finetune(model_id: str, family: str, samples: list[dict], output_dir: Path, seed: int = config.SEED,
             load_in_8bit: bool = False, max_steps: int = -1, max_length: int | None = None) -> Path:
    """Train a LoRA adapter with the configuration of Table 2; returns the adapter directory.

    `max_length` replaces config.SFT["max_length"] (needed for windows longer than the training length).
    """
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    tokenizer = load_tokenizer(model_id, family)
    dataset = Dataset.from_list([
        {"prompt": render_prompt(tokenizer, s, family), "completion": s["response"] + tokenizer.eos_token}
        for s in samples
    ])

    sft = dict(config.SFT)
    if max_length:
        sft["max_length"] = max_length
    longest = max(len(tokenizer(r["prompt"] + r["completion"]).input_ids) for r in dataset)
    if longest > sft["max_length"]:                 # TRL would silently cut the answer off the end
        raise ValueError(f"longest training sample has {longest} tokens > max_length {sft['max_length']}; "
                         "pass a larger --max-length")
    print(f"longest training sample: {longest} tokens (max_length {sft['max_length']})", flush=True)

    model_kwargs = {"dtype": torch.bfloat16}
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False

    if not torch.cuda.is_available():               # CPU runs are for testing the pipeline only
        sft.update(optim="adamw_torch", bf16=False)
    args = SFTConfig(output_dir=str(output_dir / "trainer"), seed=seed, max_steps=max_steps,
                     completion_only_loss=True, logging_steps=50, eval_strategy="no",
                     # a checkpoint every CHECKPOINT_STEPS so a crash costs minutes, not a run;
                     # only the newest is kept, and the final model is still the one after the last step
                     save_strategy="steps", save_steps=config.CHECKPOINT_STEPS, save_total_limit=1,
                     report_to="none", **sft)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
                         peft_config=LoraConfig(**config.LORA))
    wrapped = sorted({n.split(".lora_A")[0] for n, _ in trainer.model.named_parameters() if ".lora_A" in n})
    suffixes = sorted({w.rsplit(".", 1)[-1] for w in wrapped})
    print(f"LoRA wraps {len(wrapped)} modules; module types: {suffixes}", flush=True)
    trainer.model.print_trainable_parameters()
    start = time.perf_counter()
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(output_dir / "trainer")) if (output_dir / "trainer").is_dir() else None
    if last:
        print(f"resuming from {last}", flush=True)
    result = trainer.train(resume_from_checkpoint=last)
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log = {"base_model": model_id, "family": family, "seed": seed, "n_samples": len(samples),
           "lora_wrapped_modules": len(wrapped), "lora_module_types": suffixes,
           "load_in_8bit": load_in_8bit, "lora": config.LORA, "training": sft, "steps": trainer.state.global_step,
           "train_loss": result.training_loss, "wall_clock_s": round(time.perf_counter() - start, 1)}
    if torch.cuda.is_available():
        log["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        print(f"peak GPU memory: {log['peak_gpu_memory_gb']} GB; "
              f"{log['wall_clock_s'] / max(log['steps'], 1):.1f} s per step", flush=True)
    (output_dir / "training_log.json").write_text(json.dumps(log, indent=1))
    return adapter_dir


def merge_adapter(model_id: str, adapter_dir: Path, merged_dir: Path, family: str) -> Path:
    """Merge the LoRA adapter into the bf16 base weights, for serving with vLLM."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(merged_dir)
    load_tokenizer(model_id, family).save_pretrained(merged_dir)
    return Path(merged_dir)


# --------------------------------------------------------------------------- inference
def generate(model: str | Path, prompts: list[str], backend: str = "vllm",
             max_new_tokens: int = config.MAX_NEW_TOKENS, batch_size: int = 16, family: str = "qwen") -> list[str]:
    """Greedy completions of already formatted prompts."""
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=str(model), dtype="bfloat16")
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens))
        return [o.outputs[0].text for o in outputs]

    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model, family)
    tokenizer.padding_side = "left"
    cuda = torch.cuda.is_available()
    lm = AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.bfloat16 if cuda else torch.float32,
                                              device_map={"": 0} if cuda else None)
    lm.eval()
    texts = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(lm.device)
        with torch.inference_mode():
            out = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return texts


# --------------------------------------------------------------------------- label scoring
# Predictions come from the model's probabilities for the four answer strings, not from parsing free text.
# first_token_scores() is the fast path (one forward pass per prompt). It is exact up to the first token,
# which is enough when the four answers start with four different tokens (checked) and the model then
# completes them deterministically; check_scorers() measures that on a sample. sequence_scores() is the
# exact path (the full answer string plus the end token), slower, and the one to use for raw models.

def load_scorer(model: str | Path, family: str):
    """Tokenizer and bf16 model for scoring (merged fine-tuned directory, or a base model id for raw models)."""
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model, family)
    cuda = torch.cuda.is_available()
    lm = AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.bfloat16 if cuda else torch.float32,
                                              device_map={"": 0} if cuda else None)
    lm.eval()
    return tokenizer, lm


def answer_ids(tokenizer, prompt: str, labels: list[str]) -> list[list[int]]:
    """Token ids of each answer (label + "." + end token) exactly as they follow `prompt`.

    Tokenises prompt + answer jointly, the way training saw it, and checks that the prompt's own ids are
    a prefix of the joint ids, so the prompt/answer boundary is clean.
    """
    p = tokenizer(prompt).input_ids
    out = []
    for lab in labels:
        joint = tokenizer(prompt + lab + "." + tokenizer.eos_token).input_ids
        if joint[:len(p)] != p:
            raise ValueError(f"prompt and answer {lab!r} do not tokenise at a clean boundary")
        out.append(joint[len(p):])
    return out


@torch.no_grad()
def first_token_scores(tokenizer, lm, prompts: list[str], labels: list[str]) -> np.ndarray:
    """(N, 4) log-probabilities of each answer's FIRST token at the end of each prompt; batch size 1, no padding."""
    firsts = [ids[0] for ids in answer_ids(tokenizer, prompts[0], labels)]
    if len(set(firsts)) != len(firsts):
        raise ValueError("two answers share their first token; use sequence_scores()")
    out = np.empty((len(prompts), len(labels)))
    for i, prompt in enumerate(prompts):
        ids = torch.tensor([tokenizer(prompt).input_ids], device=lm.device)
        logits = lm(input_ids=ids).logits[0, -1].float()
        out[i] = logits.log_softmax(-1)[firsts].cpu().numpy()
    return out


@torch.no_grad()
def sequence_scores(tokenizer, lm, prompts: list[str], labels: list[str]) -> np.ndarray:
    """(N, 4) exact log-probabilities of each full answer (label + "." + end token); batch size 1, no padding."""
    out = np.empty((len(prompts), len(labels)))
    for i, prompt in enumerate(prompts):
        p = tokenizer(prompt).input_ids
        for j, a in enumerate(answer_ids(tokenizer, prompt, labels)):
            ids = torch.tensor([p + a], device=lm.device)
            logp = lm(input_ids=ids).logits[0, len(p) - 1:-1].float().log_softmax(-1)
            out[i, j] = logp[torch.arange(len(a)), torch.tensor(a, device=lm.device)].sum().item()
    return out


def to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Softmax over the four answers: class probabilities that sum to one per customer."""
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def check_scorers(tokenizer, lm, prompts: list[str], labels: list[str]) -> dict:
    """Agreement of the fast and the exact scorer on a sample (run on about 200 prompts per model)."""
    exact = to_probabilities(sequence_scores(tokenizer, lm, prompts, labels))
    try:
        fast = to_probabilities(first_token_scores(tokenizer, lm, prompts, labels))
    except ValueError as err:                       # two answers share their first token
        return {"n": len(prompts), "first_token_usable": False, "reason": str(err)}
    return {"n": len(prompts), "first_token_usable": True,
            "argmax_agreement": float((fast.argmax(1) == exact.argmax(1)).mean()),
            "max_abs_prob_diff": float(np.abs(fast - exact).max())}

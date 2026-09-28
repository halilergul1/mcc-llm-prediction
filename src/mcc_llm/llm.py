"""LoRA fine-tuning (Section 3.4, Table 2) and greedy inference for Qwen3.5-9B and Mistral-7B-Instruct.

Chat format: Qwen gets a system, a user and an assistant turn, with thinking disabled
in the chat template; Mistral gets a user and an assistant turn. The user turn is the
Instruction Input (Task Instruction and Task Input), the assistant turn the Instruction
Output. The loss is the log-likelihood of the Instruction Output tokens (Equation 1).

Training uses an 8-bit copy of the frozen base model (bitsandbytes) with PEFT and TRL.
For inference the adapter is merged into the bf16 base weights and served with vLLM,
greedy, at most 10 new tokens. The generated text is mapped to one of the four classes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from . import config

FAMILIES = ("qwen", "mistral")
LABELS = {"grocery": "Grocery", "clothing": "Clothing", "gas stations": "Gas stations",
          "gas station": "Gas stations", "other": "Other"}


def run_name(family: str, condition: str, drop_field: str | None = None, seed: int = config.SEED,
             customer_id: bool = True) -> str:
    """Default name of a fine-tuning run and of its prediction files, e.g. qwen-seq-seed2."""
    name = f"{family}-{condition}"
    if drop_field:
        name += f"-minus-{drop_field.replace('_', '-')}"
    if not customer_id:
        name += "-no-id"
    return name if seed == config.SEED else f"{name}-seed{seed}"


# --------------------------------------------------------------------------- prompts and parsing
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


def chat_messages(sample: dict, family: str) -> list[dict]:
    messages = [] if family == "mistral" else [{"role": "system", "content": sample["system"]}]
    messages.append({"role": "user", "content": f"{sample['instruction']}\n\n{sample['input']}"})
    return messages


def render_prompt(tokenizer, sample: dict, family: str) -> str:
    """Chat-formatted prompt ending where the answer starts. Used for training and inference alike."""
    extra = {"enable_thinking": False} if family == "qwen" else {}
    text = tokenizer.apply_chat_template(chat_messages(sample, family), tokenize=False,
                                         add_generation_prompt=True, **extra)
    if family == "qwen" and "<think>" in (tokenizer.chat_template or "") and not text.endswith("</think>\n\n"):
        text += "<think>\n\n</think>\n\n"          # thinking disabled: empty reasoning block
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]     # the tokenizer adds BOS itself
    return text


def parse_category(text: str) -> str | None:
    """Map generated text to a class: prefix match first, then substring match."""
    t = text.strip().rstrip(".").lower()
    for key, label in LABELS.items():
        if t.startswith(key):
            return label
    for key, label in LABELS.items():
        if key in t:
            return label
    return None


# --------------------------------------------------------------------------- fine-tuning
def finetune(model_id: str, family: str, samples: list[dict], output_dir: Path, seed: int = config.SEED,
             load_in_8bit: bool = True, max_steps: int = -1) -> Path:
    """Train a LoRA adapter with the configuration of Table 2; returns the adapter directory."""
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

    model_kwargs = {"dtype": torch.bfloat16}
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.config.use_cache = False

    sft = dict(config.SFT)
    if not torch.cuda.is_available():               # CPU runs are for testing the pipeline only
        sft.update(optim="adamw_torch", bf16=False)
    args = SFTConfig(output_dir=str(output_dir / "trainer"), seed=seed, max_steps=max_steps,
                     completion_only_loss=True, logging_steps=50, save_strategy="no", eval_strategy="no",
                     report_to="none", **sft)
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
                         peft_config=LoraConfig(**config.LORA))
    start = time.perf_counter()
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    log = {"base_model": model_id, "family": family, "seed": seed, "n_samples": len(samples),
           "load_in_8bit": load_in_8bit, "lora": config.LORA, "training": sft, "steps": trainer.state.global_step,
           "train_loss": result.training_loss, "wall_clock_s": round(time.perf_counter() - start, 1)}
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

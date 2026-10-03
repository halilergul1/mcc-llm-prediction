"""Instruction template (Section 3.3, Table 1) and the prompt conditions of the ablations (Section 3.7).

A sample has four parts: the Task Instruction, the Task Input (the customer's profile and
history as a short first-person narrative), their concatenation, and the Instruction Output,
the category of the target transaction.

  * One template for every condition: the sentences always come in the order demographics,
    categories, dates, amounts, and a condition only leaves out its absent sentences.
  * One instruction for every condition, ending with the label set, so raw and fine-tuned models
    see the same prompt and a raw model knows the valid answers.
  * No customer identifier in any condition.
  * Amounts are printed in their currency (config.CURRENCY).
  * The `neutral` condition replaces the category names by A/B/C/D (inputs, label set, answer).

Conditions:
    full                  Q-full: demographics, categories, dates and amounts
    seq                   Q-seq: the chronological category sequence only
    seq+<fields>          the category sequence plus any combination of amounts, dates and
                          demographics (the 2^3 factorial; full is the corner with all three)
    neutral               Q-neutral: seq with neutral tokens instead of the category names
Leave-one-field-out removes one demographic field from `full`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import config
from .data import Windows

INSTRUCTION = "Based on the information about me below, predict the category of my next purchase. Answer with one of: {labels}."
NEUTRAL_TOKENS = dict(zip(config.CATEGORIES, "ABCD"))     # class code 0 is A, 1 is B, 2 is C, 3 is D
DEMOGRAPHIC_FIELDS = ("gender", "age", "marital_status", "education", "occupation", "income_group")


@dataclass(frozen=True)
class Context:
    """Fields the Task Input contains in addition to the category sequence."""

    amounts: bool = True
    dates: bool = True
    demographics: bool = True
    neutral: bool = False           # category names replaced by NEUTRAL_TOKENS
    drop_field: str | None = None   # demographic field left out (leave-one-field-out)

    @property
    def sequence_only(self) -> bool:
        return not (self.amounts or self.dates or self.demographics)


CONDITIONS = {
    "full": Context(),
    "seq": Context(False, False, False),
    "seq+amounts": Context(True, False, False),
    "seq+dates": Context(False, True, False),
    "seq+demographics": Context(False, False, True),
    "seq+amounts+dates": Context(True, True, False),
    "seq+amounts+demographics": Context(True, False, True),
    "seq+dates+demographics": Context(False, True, True),
    "neutral": Context(False, False, False, neutral=True),
}


def get_context(condition: str, drop_field: str | None = None) -> Context:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {list(CONDITIONS)}")
    if drop_field is not None and drop_field not in DEMOGRAPHIC_FIELDS:
        raise ValueError(f"unknown demographic field {drop_field!r}; choose from {DEMOGRAPHIC_FIELDS}")
    return replace(CONDITIONS[condition], drop_field=drop_field)


def label_words(ctx: Context) -> list[str]:
    """The answer strings in class-code order (config.CATEGORIES), as the model must write them."""
    return [NEUTRAL_TOKENS[c] if ctx.neutral else c for c in config.CATEGORIES]


def instruction(ctx: Context) -> str:
    return INSTRUCTION.format(labels=", ".join(label_words(ctx)))


def money(value: float) -> str:
    return f"{config.CURRENCY} {float(value):.2f}"


def demographic_sentences(profile, drop_field: str | None = None) -> str:
    """'I am 49 years old, married male, high school graduate, and private sector employee. In terms of ...'"""
    parts = []
    if drop_field != "age":
        parts.append(f"{int(profile['age'])} years old")
    person = " ".join(str(profile[f]) for f in ("marital_status", "gender") if f != drop_field)
    if person:
        parts.append(person)
    if drop_field != "education":
        parts.append(f"{profile['education']} graduate")
    if drop_field != "occupation":
        parts.append(str(profile["occupation"]))
    text = "I am " + (", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 1 else parts[0]) + "."
    if drop_field != "income_group":
        text += f" In terms of my income, I belong to the {profile['income_group']} income group."
    return text


def task_input(ctx: Context, categories, dates, amounts, profile=None) -> str:
    """Render the Task Input for one window of input transactions (oldest first), in the fixed sentence order."""
    words = [NEUTRAL_TOKENS[str(c)] if ctx.neutral else str(c) for c in categories]
    sentences = []
    if ctx.demographics and profile is not None:
        sentences.append(demographic_sentences(profile, ctx.drop_field))
    sentences.append(f"I made {len(words)} purchases. Their categories, oldest first: {', '.join(words)}.")
    if ctx.dates:
        sentences.append("Their dates, oldest first: "
                         + ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates) + ".")
    if ctx.amounts:
        sentences.append("Their amounts, oldest first: " + ", ".join(money(a) for a in amounts)
                         + f". In total I spent {money(round(float(sum(amounts)), 2))}.")
    return " ".join(sentences)


def build_samples(windows: Windows, demographics: pd.DataFrame | None, ctx: Context) -> list[dict]:
    """One instruction sample per customer. `response` is the Instruction Output, e.g. 'Clothing.'."""
    if ctx.demographics and demographics is None:
        raise ValueError("this condition needs demographics, but none were loaded")
    words = dict(zip(config.CATEGORIES, label_words(ctx)))
    samples = []
    for i, cid in enumerate(windows.customer_id):
        profile = None
        if ctx.demographics:
            if cid not in demographics.index:
                raise KeyError(f"customer {cid} has no demographics")
            profile = demographics.loc[cid]
        samples.append({
            "customer_id": cid,
            "instruction": instruction(ctx),
            "input": task_input(ctx, windows.categories[i, :-1], windows.dates[i, :-1], windows.amounts[i, :-1], profile),
            "response": f"{words[windows.target[i]]}.",
            "target_class": config.CATEGORIES.index(windows.target[i]),
        })
    return samples


def few_shot_examples(train_samples: list[dict], per_class: int = config.FEW_SHOT_PER_CLASS,
                      seed: int = config.SEED) -> list[dict]:
    """A fixed, class-balanced set of training samples for the raw models' few-shot prompts."""
    rng = np.random.default_rng(seed)
    chosen = []
    for k in range(config.NUM_CLASSES):
        pool = [s for s in train_samples if s["target_class"] == k]
        if len(pool) < per_class:
            raise ValueError(f"only {len(pool)} training samples of class {config.CATEGORIES[k]}")
        chosen += [pool[j] for j in rng.choice(len(pool), size=per_class, replace=False)]
    order = rng.permutation(len(chosen))                      # classes interleaved, same order for every customer
    return [chosen[j] for j in order]


def with_few_shot(sample: dict, examples: list[dict]) -> dict:
    """Copy of `sample` whose Task Input is preceded by the worked examples."""
    shots = "\n\n".join(f"Example {i + 1}:\n{e['input']}\nAnswer: {e['response']}" for i, e in enumerate(examples))
    return {**sample, "input": f"Examples from other customers:\n\n{shots}\n\nNow my information:\n{sample['input']}"}

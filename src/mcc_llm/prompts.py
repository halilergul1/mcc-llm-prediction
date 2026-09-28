"""Instruction template (Section 3.3, Table 1) and the prompt conditions of the ablations (Section 3.7).

A sample has four parts: the Task Instruction, the Task Input (the customer's
profile and history as a short first-person narrative), their concatenation, and
the Instruction Output, the category of the target transaction.

Conditions:
    full                  Q-full: identifier, demographics, amounts, dates and categories (Table 1)
    seq                   Q-seq: the chronological category sequence only
    seq+<fields>          the category sequence plus any combination of amounts, dates and
                          demographics (the 2^3 factorial; full is the corner with all three)
Leave-one-field-out removes one demographic field from `full`; the identifier
sentence can be removed at inference to test whether it carries information.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from .data import Windows

INSTRUCTION_FULL = (
    "Based on my demographic details and historical transaction data provided below, "
    "predict my next purchase category."
)
INSTRUCTION_SEQ = (
    "Based on the following chronological sequence of purchase categories, predict the next purchase category."
)
DEMOGRAPHIC_FIELDS = ("gender", "age", "marital_status", "education", "occupation", "income_group")


@dataclass(frozen=True)
class Context:
    """Fields the Task Input contains in addition to the category sequence."""

    amounts: bool = True
    dates: bool = True
    demographics: bool = True
    customer_id: bool = True
    drop_field: str | None = None  # demographic field left out (leave-one-field-out)

    @property
    def sequence_only(self) -> bool:
        return not (self.amounts or self.dates or self.demographics)

    @property
    def full(self) -> bool:
        return self.amounts and self.dates and self.demographics


CONDITIONS = {
    "full": Context(),
    "seq": Context(False, False, False, False),
    "seq+amounts": Context(True, False, False, False),
    "seq+dates": Context(False, True, False, False),
    "seq+demographics": Context(False, False, True, False),
    "seq+amounts+dates": Context(True, True, False, False),
    "seq+amounts+demographics": Context(True, False, True, False),
    "seq+dates+demographics": Context(False, True, True, False),
}


def get_context(condition: str, drop_field: str | None = None, customer_id: bool | None = None) -> Context:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {list(CONDITIONS)}")
    if drop_field is not None and drop_field not in DEMOGRAPHIC_FIELDS:
        raise ValueError(f"unknown demographic field {drop_field!r}; choose from {DEMOGRAPHIC_FIELDS}")
    ctx = replace(CONDITIONS[condition], drop_field=drop_field)
    return ctx if customer_id is None else replace(ctx, customer_id=customer_id)


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
    text = "I am " + (", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 1 else parts[0]) + ". "
    if drop_field != "income_group":
        text += f"In terms of my income state, I belong to the {profile['income_group']} income group. "
    return text


def task_input(ctx: Context, customer_id, categories, dates, amounts, profile=None) -> str:
    """Render the Task Input for one window of input transactions (oldest first)."""
    n = len(categories)
    cats = ", ".join(str(c) for c in categories)
    date_list = ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates)
    amount_list = ", ".join(f"${float(a):.2f}" for a in amounts)
    total = round(float(sum(amounts)), 2)
    ident = f"I am {customer_id}. " if ctx.customer_id else ""
    demo = demographic_sentences(profile, ctx.drop_field) if ctx.demographics and profile is not None else ""
    if ctx.full:
        return (
            f"{ident}{demo}Recently, I made {n} transactions. "
            f"In these transactions, I have spent a total of ${total} dollars. "
            f"I bought items from the following categories, chronologically: {cats}. "
            f"I bought from these categories on the following dates, chronologically: {date_list}. "
            f"I spent the following money for these items, chronologically: {amount_list}."
        )
    text = f"{ident}{demo}I made {n} purchases from the following categories, chronologically: {cats}."
    if ctx.dates:
        text += f" I bought from these categories on the following dates, chronologically: {date_list}."
    if ctx.amounts:
        text += (
            f" In these purchases, I have spent a total of ${total} dollars."
            f" I spent the following money for these items, chronologically: {amount_list}."
        )
    return text


def build_samples(windows: Windows, demographics: pd.DataFrame | None, ctx: Context) -> list[dict]:
    """One instruction sample per customer. `response` is the Instruction Output, e.g. 'Grocery.'."""
    instruction = INSTRUCTION_SEQ if ctx.sequence_only else INSTRUCTION_FULL
    samples = []
    for i, cid in enumerate(windows.customer_id):
        profile = None
        if demographics is not None and cid in demographics.index:
            profile = demographics.loc[cid]
        samples.append({
            "customer_id": cid,
            "instruction": instruction,
            "input": task_input(ctx, cid, windows.categories[i, :-1], windows.dates[i, :-1],
                                windows.amounts[i, :-1], profile),
            "response": f"{windows.target[i]}.",
        })
    return samples

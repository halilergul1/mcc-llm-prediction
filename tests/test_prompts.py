import pandas as pd

from mcc_llm import prompts

PROFILE = pd.Series({"age": 49, "gender": "male", "marital_status": "married", "education": "high school",
                     "occupation": "private sector employee", "income_group": "low"})
CATS = ["Grocery", "Grocery", "Other", "Other", "Grocery", "Grocery", "Other", "Other", "Other"]
DATES = pd.to_datetime(["2013-04-14", "2013-04-15", "2013-04-19", "2013-04-21", "2013-05-18",
                        "2013-05-18", "2013-06-16", "2013-06-17", "2013-06-19"])
AMOUNTS = [11.27, 2.00, 25.70, 20.00, 300.00, 100.00, 50.00, 300.00, 65.00]


def test_full_template_matches_table_1():
    text = prompts.task_input(prompts.CONDITIONS["full"], 1695432, CATS, DATES, AMOUNTS, PROFILE)
    assert text == (
        "I am 1695432. I am 49 years old, married male, high school graduate, and private sector employee. "
        "In terms of my income state, I belong to the low income group. Recently, I made 9 transactions. "
        "In these transactions, I have spent a total of $873.97 dollars. I bought items from the following "
        "categories, chronologically: Grocery, Grocery, Other, Other, Grocery, Grocery, Other, Other, Other. "
        "I bought from these categories on the following dates, chronologically: 2013-04-14, 2013-04-15, "
        "2013-04-19, 2013-04-21, 2013-05-18, 2013-05-18, 2013-06-16, 2013-06-17, 2013-06-19. I spent the "
        "following money for these items, chronologically: $11.27, $2.00, $25.70, $20.00, $300.00, $100.00, "
        "$50.00, $300.00, $65.00."
    )


def test_sequence_only_template():
    text = prompts.task_input(prompts.CONDITIONS["seq"], 1695432, CATS, DATES, AMOUNTS, PROFILE)
    assert text == ("I made 9 purchases from the following categories, chronologically: "
                    "Grocery, Grocery, Other, Other, Grocery, Grocery, Other, Other, Other.")


def test_leave_one_field_out_and_identifier():
    assert prompts.demographic_sentences(PROFILE, "gender").startswith(
        "I am 49 years old, married, high school graduate, and private sector employee.")
    assert "income group" not in prompts.demographic_sentences(PROFILE, "income_group")
    no_id = prompts.get_context("full", customer_id=False)
    assert prompts.task_input(no_id, 1695432, CATS, DATES, AMOUNTS, PROFILE).startswith("I am 49 years old")

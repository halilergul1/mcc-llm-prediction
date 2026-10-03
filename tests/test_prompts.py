import pandas as pd

from mcc_llm import config, prompts

PROFILE = pd.Series({"age": 49, "gender": "male", "marital_status": "married", "education": "high school",
                     "occupation": "private sector employee", "income_group": "low"})
F = "Food and grocery"
CATS = [F, F, "Other", "Other", F, F, "Other", "Other", "Other"]
DATES = pd.to_datetime(["2013-04-14", "2013-04-15", "2013-04-19", "2013-04-21", "2013-05-18",
                        "2013-05-18", "2013-06-16", "2013-06-17", "2013-06-19"])
AMOUNTS = [11.27, 2.00, 25.70, 20.00, 300.00, 100.00, 50.00, 300.00, 65.00]


def test_full_template_v3():
    text = prompts.task_input(prompts.CONDITIONS["full"], CATS, DATES, AMOUNTS, PROFILE)
    assert text == (
        "I am 49 years old, married male, high school graduate, and private sector employee. "
        "In terms of my income, I belong to the low income group. "
        "I made 9 purchases. Their categories, oldest first: Food and grocery, Food and grocery, Other, Other, "
        "Food and grocery, Food and grocery, Other, Other, Other. "
        "Their dates, oldest first: 2013-04-14, 2013-04-15, 2013-04-19, 2013-04-21, "
        "2013-05-18, 2013-05-18, 2013-06-16, 2013-06-17, 2013-06-19. Their amounts, oldest first: TL 11.27, "
        "TL 2.00, TL 25.70, TL 20.00, TL 300.00, TL 100.00, TL 50.00, TL 300.00, TL 65.00. In total I spent TL 873.97."
    )
    assert "$" not in text and "dollar" not in text and "I am 1" not in text


def test_every_cell_is_the_full_template_minus_its_absent_sentences():
    full = prompts.task_input(prompts.CONDITIONS["full"], CATS, DATES, AMOUNTS, PROFILE)
    for name, ctx in prompts.CONDITIONS.items():
        if ctx.neutral:
            continue
        text = prompts.task_input(ctx, CATS, DATES, AMOUNTS, PROFILE)
        for sentence in text.split(". "):
            assert sentence.rstrip(".") in full, (name, sentence)


def test_one_instruction_with_the_label_set():
    seq, full = prompts.instruction(prompts.CONDITIONS["seq"]), prompts.instruction(prompts.CONDITIONS["full"])
    assert seq == full and seq.endswith("Answer with one of: Clothing, Gas stations, Food and grocery, Other.")
    assert prompts.instruction(prompts.CONDITIONS["neutral"]).endswith("Answer with one of: A, B, C, D.")


def test_neutral_condition_hides_the_names():
    text = prompts.task_input(prompts.CONDITIONS["neutral"], CATS, DATES, AMOUNTS, PROFILE)
    assert text == "I made 9 purchases. Their categories, oldest first: C, C, D, D, C, C, D, D, D."
    assert prompts.label_words(prompts.CONDITIONS["neutral"]) == ["A", "B", "C", "D"]
    assert prompts.label_words(prompts.CONDITIONS["full"]) == list(config.CATEGORIES)


def test_leave_one_field_out():
    assert prompts.demographic_sentences(PROFILE, "gender").startswith(
        "I am 49 years old, married, high school graduate, and private sector employee.")
    assert "income group" not in prompts.demographic_sentences(PROFILE, "income_group")

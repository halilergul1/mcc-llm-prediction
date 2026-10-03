# Analysis plan (pre-registered) — manuscript 9171855, revision v3

This plan was committed on 2026-10-03, before the training runs of this revision; only tests of the pipeline
on the validation, in-bank and development splits preceded it. The settings of every model, the LLM runs
included, are fixed by the pipeline commit below. The non-LLM models are scored on every split, the
confirmation splits included, when they are trained. The LLMs are scored on the confirmation splits once,
after the freeze: the tag `v3-prereg` is set together with the manifest of the trained models (section 6)
BEFORE any LLM is scored on `bank_b_confirm`. The paper and the response letter cite the tagged commit and
its date. Nothing below may change after the first scoring of a confirmation split; any later analysis is
reported as exploratory.

- Pipeline commit: `670da5e` · Plan date: 2026-10-03 · Sign-off by SB, HIE and BB: recorded with the tag `v3-prereg`
- Amended on 2026-10-03, before any model was scored on a confirmation split: the non-LLM models are scored on
  the confirmation splits when they are trained, not after the freeze (section 6). The first version of this
  plan is commit `77108bc`.

## 1. Data

- Pipeline: the commit above, `config.py` as committed.
  - Labels (ISO 18245 merchant category codes): Gas stations 5541, 5542, 5983; Clothing 5600-5699, 5137,
    5139, 5941; Food and grocery 5400-5499; every other code is Other.
  - Amounts: nominal Turkish lira, printed as "TL 300.00"; no deflation.
  - Eligible customers: complete demographics, at least 10 transactions, at least 2 distinct categories
    among the 9 transactions before the target. The filter does not use the target.
  - Transactions of one day are ordered by their recorded time; a transaction without a recorded time comes
    first on its day.
  - The demographic labels of the two banks are mapped to one vocabulary (`scripts/export_raw.py`).
- Splits (`scripts/prepare_data.py`, seed 42; checksums in `results/MANIFEST_data`): train 40,000, val 10,000,
  bank_a_test 10,000, bank_b_dev 971, bank_b_confirm 10,257, bank_b_unfiltered 14,059. At length 14,
  bank_b_confirm has 6,514 customers (those with at least 15 transactions).
- Excluded from bank_b_confirm and bank_b_unfiltered: the 1,000 Bank B test customers of the earlier
  evaluations (`old/earlier_test_customers.csv`, SHA-256 `bb0e9a960adab7d0…`). The 971 of them that are
  eligible under the rule above form bank_b_dev.
- bank_b_dev is used for smoke tests only. No modelling decision uses it.

## 2. Primary endpoint

- Metric: macro-F1 at input length 9 on `bank_b_confirm`.
- Comparison: Q-full (mean over seeds 42, 2, 3) minus the non-LLM model with the highest macro-F1 on the
  validation split (among averaging, last-category, markov1, lstm, cnn, sasrec, dros-sasrec, gbdt-seq,
  gbdt-full, logit-seq, logit-full; seed means where seeds exist).
- Predictions: label scoring. The prediction of an LLM is the most probable of the four answers under the
  model (`scripts/predict_llm.py`); free generation is a check only.
- Test: paired bootstrap over customers, 10,000 resamples, seed 42, two-sided, alpha = 0.05. No multiplicity
  correction for this single test. Command: `python scripts/evaluate_v3.py primary --llm qwen-full`.

## 3. Secondary endpoints (all labelled secondary in the paper)

- Accuracy (exact McNemar), weighted F1, balanced accuracy and class-wise F1 at lengths 4, 7, 9, 14;
  PR-AUC per class, top-2 accuracy, ECE and Brier score.
- Holm families: per LLM (Q-full, Q-seq, Mistral-full), per metric, per length, over the non-LLM comparators
  listed in section 2. Command: `python scripts/evaluate_v3.py families` (seed means);
  `python scripts/evaluate.py --predictions results/predictions/bank_b_confirm` gives the tables and tests of
  the single runs.
- Ablations (single paired comparisons, unadjusted, reported with their MDE): Q-full − Q-seq (seed means);
  Q-seq − Q-neutral; factorial main effects and interactions; leave-one-field-out.
- In-bank against cross-bank: every model at length 9 on bank_a_test and bank_b_confirm (`evaluate_v3.py shift`).
- Subgroups: gender and age band (`evaluate_v3.py subgroups`). Descriptive, no tests.
- Operating point: Clothing threshold chosen on validation with cost FN : FP = 5 : 1, applied unchanged
  to bank_b_confirm (`evaluate_v3.py calibration`).

## 4. Sensitivity analyses (pre-declared)

- bank_b_unfiltered: the primary comparison without any diversity filter
  (`evaluate_v3.py primary --split bank_b_unfiltered`).
- Tied target dates: the primary comparison without customers whose target date holds two or more
  transactions (`evaluate_v3.py primary --without-ties`).

## 5. Reporting rules

- Main tables: seed means ± SD, three decimals. The minimum detectable effect (`Diff … MDE`) beside every
  non-significant difference.
- No customer is excluded after scoring. No model, prompt or hyperparameter changes after the first scoring of
  a confirmation split; a change forced by a technical failure is reported in the paper as a deviation.

## 6. Freeze

- Before training: this plan and `results/MANIFEST_data` are committed.
- Non-LLM models: scored on every split when they are trained (`scripts/train_baselines.sh`). Their grids,
  seeds and selection rule are fixed by the pipeline commit, and the comparator of the primary endpoint is
  chosen on the validation split (section 2).
- LLMs: until the freeze, scored only on val, bank_a_test and bank_b_dev.
- After training, before any LLM is scored on a confirmation split: `python scripts/manifest.py runs --out
  results/MANIFEST_models` → commit the file and tag the commit `v3-prereg`.
- After scoring (`scripts/score_all.sh confirm`): `python scripts/manifest.py results/predictions --out
  results/MANIFEST_predictions` → commit the file.

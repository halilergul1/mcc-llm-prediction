# mcc-llm-prediction

Code for *Instruction-Tuned Large Language Models for Cross-Institution Merchant Category Prediction*
(Ergul, Balcisoy and Bozkaya, Expert Systems, under review).

The task is to predict the category of a customer's next card purchase (Clothing, Gas stations, Food and
grocery, or Other) from the purchase history. Models are trained on the customers of one bank (Bank A) and
tested on held-out customers of the same bank and on the customers of another bank (Bank B). Every model
uses the same per-customer protocol: the customer's last X transactions are the input and the (X+1)-th,
the most recent one, is the target. Each customer gives one sample; there is no sliding window. Training
uses X = 9, and evaluation uses X = 4, 7, 9 and 14.

The bank data cannot be shared. The scripts run on any data in the schema below, and
`scripts/make_example_data.py` writes synthetic data to try them. The analyses of the paper are fixed in
advance in [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).

## Layout

```
src/mcc_llm/
  config.py        every setting: labels, cohort filters, splits, baseline and LoRA hyperparameters
  data.py          input schema, cohort filters, customer draws, per-customer X+1 windows
  prompts.py       instruction template and prompt conditions (full, seq, factorial cells, neutral)
  baselines.py     Random, Proportional, Averaging, last-category, first-order Markov chain
  features.py      feature tables, LightGBM and logistic regression
  models.py        LSTM, CNN, SASRec, DROS-SASRec
  training.py      baseline training loops, grid search, seeds, saved models
  llm.py           LoRA fine-tuning, adapter merging, label scoring, free generation
  evaluation.py    metrics, paired bootstrap, exact McNemar test, Holm correction, probability metrics
  predictions.py   per-customer prediction files
scripts/           one entry point per step; run_v3.sh lists them in order
tests/             unit tests (pytest)
examples/          example of an export spec
```

## Data

| File | Columns |
|---|---|
| transactions (CSV, or a folder of CSVs) | `customer_id`, `date`, `amount`, `mcc` (four-digit merchant category code), optionally `time` |
| demographics (CSV) | `customer_id`, `age`, `gender`, `marital_status`, `education`, `occupation`, `income` |

`scripts/export_raw.py` writes these two files from a bank's raw files. A small JSON spec per bank names
the raw files and columns and gives one word for every raw demographic label, so that both banks share one
vocabulary ([examples/export_spec.example.json](examples/export_spec.example.json)). Transactions of one
day are ordered by `time`; a transaction without a recorded time comes first on its day. Amounts stay in
their currency (nominal Turkish lira in the paper). Income becomes a within-bank tercile (low, middle,
high), computed over all complete demographic records.

**Labels** (`config.MCC_CATEGORIES`, ISO 18245 codes): Gas stations 5541, 5542, 5983; Clothing 5600-5699,
5137, 5139, 5941; Food and grocery 5400-5499; every other code, eating places (5811-5814) included, is
Other.

**Cohort.** A customer is eligible with complete demographics, at least 10 transactions, and at least 2
distinct categories among the 9 transactions before the target. The filter never looks at the target.

**Splits** (`scripts/prepare_data.py`, seed 42):

| Split | Customers | Use |
|---|---|---|
| `train`, `val` | 40,000 and 10,000 eligible Bank A customers | training; every tuning and selection decision |
| `bank_a_test` | 10,000 other eligible Bank A customers | in-bank reference |
| `bank_b_dev` | the test customers of earlier evaluations | checks only; no modelling decision |
| `bank_b_confirm` | every other eligible Bank B customer | the primary evaluation; the LLMs are scored on it once, after the freeze |
| `bank_b_unfiltered` | Bank B customers with 10 or more transactions, without the category filter | sensitivity analysis |

## Models

| Model | Script | Notes |
|---|---|---|
| Random, Proportional, Averaging, last-category, Markov-1 | `run_baselines.py` | Averaging ties go to the class seen most recently |
| LightGBM, logistic regression | `train_features.py` | feature sets `seq` (category sequence) and `full` (plus amounts, dates, demographics) |
| LSTM, CNN, SASRec, DROS-SASRec | `train_lstm.py`, `train_cnn.py`, `train_sasrec.py`, `train_dros.py` | class embeddings; grid search on `val`, weighted or unweighted loss, 5 seeds |
| Qwen3.5-9B, Mistral-7B-Instruct-v0.3 with LoRA | `train_qwen.py`, `train_mistral.py`, `predict_llm.py` | conditions below; seeds 42, 2, 3 |

LLM prompt conditions (`--condition`): `full` (demographics, categories, dates, amounts), `seq` (categories
only), `seq+amounts`, `seq+dates`, `seq+demographics` and their pairs (a 2^3 factorial with `seq` and
`full` as corners), and `neutral` (`seq` with A/B/C/D in place of the category names). `--drop-field`
removes one demographic field from `full`. Every condition uses one template and one instruction, which
ends with the label set; no prompt contains a customer identifier.

An LLM predicts by label scoring: the model's probabilities of the four answers are computed, the most
probable one is the prediction, and the four probabilities are stored. Raw models are scored the same way,
zero-shot and with 16 fixed training examples. Free generation is used only to count answers that are not
a category.

Hyperparameters are selected on `val` by macro-F1. The baseline scripts save every model and score every
split; `--score-only` scores saved models again without training.

## Install

```bash
pip install -e .              # data preparation, baselines, evaluation
pip install -e ".[llm]"       # LoRA fine-tuning and label scoring (CUDA GPU)
pip install -e ".[serve]"     # vLLM, for the free-generation check only
pip install -e ".[dev]"       # pytest
git clone https://github.com/YangZhengyi98/DROS third_party/DROS && git -C third_party/DROS checkout 44ff99d
```

The official DROS code has no licence file, so it is not included here; `train_dros.py` imports it from
`third_party/DROS`.

## Run

`scripts/run_v3.sh` holds the full run order. In short:

```bash
# 0. raw bank files -> input schema (where the bank data is)
python scripts/export_raw.py --spec specs/bank_a.json --out raw/bank_a
python scripts/export_raw.py --spec specs/bank_b.json --out raw/bank_b

# 1. cohorts and splits; record their checksums
python scripts/prepare_data.py \
    --train-transactions raw/bank_a/transactions.csv --train-demographics raw/bank_a/demographics.csv \
    --test-transactions raw/bank_b/transactions.csv --test-demographics raw/bank_b/demographics.csv \
    --dev-ids old/earlier_test_customers.csv --exclude old/earlier_test_customers.csv
python scripts/manifest.py data/processed --out results/MANIFEST_data

# 2. non-LLM models: train, save, and score every split
scripts/train_baselines.sh          # run_baselines.py, train_features.py, train_lstm.py, train_cnn.py, train_sasrec.py, train_dros.py

# 3. LLM fine-tuning queue (resumable; one job at a time)
mkdir -p logs && nohup scripts/queue.sh queue_5090.txt > logs/queue_5090.out 2>&1 &

# 4. LLM scoring allowed before the freeze
scripts/score_all.sh pre            # LLMs on val, bank_a_test, bank_b_dev
scripts/score_all.sh generation     # free generation on bank_b_dev (vLLM, or GEN_BACKEND=transformers)

# 5. freeze, then the LLMs on the confirmation splits, once
python scripts/manifest.py runs --out results/MANIFEST_models       # commit it, tag v3-prereg
scripts/score_all.sh confirm        # the LLMs on bank_b_confirm and bank_b_unfiltered
python scripts/manifest.py results/predictions --out results/MANIFEST_predictions

# 6. evaluation
python scripts/evaluate_v3.py primary --llm qwen-full
python scripts/evaluate.py --predictions results/predictions/bank_b_confirm --out results/metrics/bank_b_confirm
python scripts/evaluate_v3.py summary --split bank_b_confirm
python scripts/evaluate_v3.py families && python scripts/evaluate_v3.py shift
python scripts/evaluate_v3.py subgroups && python scripts/evaluate_v3.py calibration --model qwen-full
```

To try the pipeline without bank data:

```bash
python scripts/make_example_data.py --out data/example
python scripts/prepare_data.py --train-transactions data/example/bank_a/transactions.csv \
    --train-demographics data/example/bank_a/demographics.csv \
    --test-transactions data/example/bank_b/transactions.csv \
    --test-demographics data/example/bank_b/demographics.csv --n-train-pool 2000 --n-bank-a-test 400 \
    --dev-ids data/example/bank_b/earlier_test_customers.csv --exclude data/example/bank_b/earlier_test_customers.csv
```

**On a GPU server.** Clone the repository, install it, clone DROS, and copy `data/processed/` from the
machine that prepared it. `python scripts/manifest.py data/processed --check results/MANIFEST_data`
confirms that the copy is complete. Steps 2 to 6 need nothing else. Save the environment with
`pip freeze > requirements.lock`.

## Outputs

`runs/` holds the saved baseline models, the LoRA adapters and the training logs. `results/predictions/<split>/`
holds one CSV per model and length with `customer_id`, `ground_truth`, `prediction` and the class
probabilities `p_<class>`. Seed replicates are named `<model>-seed<k>`.

| Output | Content |
|---|---|
| `results/metrics_v3/primary_endpoint.csv` | the pre-registered test: macro-F1 at length 9 on `bank_b_confirm`, LLM seed mean against the best non-LLM model on `val` |
| `results/metrics/<split>/` | `metrics.csv` (with 95% intervals), `confusion.csv`, `significance.csv` (paired bootstrap, exact McNemar, Holm), `pairs.csv`, `factorial.csv` |
| `results/metrics_v3/<split>/` | `summary.csv` and `summary_seed_means.csv` (PR-AUC, top-2 accuracy, ECE, Brier), `significance_seed_means.csv` |
| `results/metrics_v3/` | `shift_in_vs_cross_bank.csv`, `subgroups_<split>.csv`, `reliability_<model>_<split>.csv` |

Every paired comparison reports its difference, 95% interval, p-value and the smallest difference it could
have detected (`MDE`).

## Settings

All values are in `src/mcc_llm/config.py`.

| | |
|---|---|
| LoRA | r 32, alpha 64, dropout 0.05; q, k, v, o, gate, up and down projections |
| Base weights | bf16, frozen, no quantisation |
| Optimisation | paged AdamW 8-bit, learning rate 2e-4, cosine decay, 100 warm-up steps, weight decay 0.01, gradient clipping 0.3 |
| Batch and epochs | 2 per device x 8 accumulation steps (16); 3 epochs, final checkpoint, no early stopping |
| Other | bf16, maximum length 512 tokens, seeds 42, 2, 3 |
| Prediction | adapter merged into bf16 weights; label scoring over the four answers |
| Baselines | grid search on `val` by macro-F1; loss weighting in the grid; seeds 42, 2, 3, 4, 5 |
| Evaluation | 10,000 paired bootstrap resamples (seed 42), exact McNemar test, Holm over the non-LLM comparators |

## Citation

```bibtex
@article{ergul2026instruction,
  title   = {Instruction-Tuned Large Language Models for Cross-Institution Merchant Category Prediction},
  author  = {Ergul, Halil Ibrahim and Balcisoy, Selim and Bozkaya, Burcin},
  journal = {Expert Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT; see `LICENSE` for third-party notes.

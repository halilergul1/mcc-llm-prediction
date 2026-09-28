# mcc-llm-prediction

Code for *Instruction-Tuned Large Language Models for Cross-Institution Merchant Category Prediction*
(Ergul, Balcisoy and Bozkaya, Expert Systems, under review).

The task is to predict a customer's next merchant category (Clothing, Gas stations, Grocery or Other)
from the transaction history. Models are trained on the customers of one bank and tested on the customers
of another. Every model uses the same per-customer protocol: the customer's last X transactions are the
input and the (X+1)-th, the most recent one, is the target. Each customer gives one sample; there is no
sliding window. Training uses X = 9, and evaluation uses X = 4, 7, 9 and 14.

The bank data used in the paper cannot be shared. The scripts run on any data in the schema below.
`scripts/make_example_data.py` writes synthetic data to try them.

## Layout

```
src/mcc_llm/
  config.py        settings of the paper: cohort filters, lengths, baseline and LoRA hyperparameters
  data.py          input schema, cohort filters, customer draws, per-customer X+1 windows
  prompts.py       instruction template (Table 1) and ablation conditions
  baselines.py     Random, Proportional, Averaging
  models.py        LSTM, CNN, SASRec, DROS-SASRec
  training.py      baseline training loops and grid search on the validation split
  llm.py           LoRA fine-tuning, adapter merging, vLLM inference, output parsing
  evaluation.py    metrics, paired bootstrap, exact McNemar test, Holm correction
  predictions.py   per-customer prediction files
scripts/           one entry point per step (below)
tests/             unit tests (pytest)
```

## Data schema

| File | Columns |
|---|---|
| transactions (CSV, or a folder of CSVs) | `customer_id`, `date`, `amount`, `mcc` (four-digit merchant category code) |
| demographics (CSV, optional) | `customer_id`, `age`, `gender`, `marital_status`, `education`, `occupation`, `income` |

The MCC-to-class map is in `config.py` (`--mcc-map` replaces it). Income becomes a within-bank tercile
(low, middle, high). Text fields enter the prompt in lower case, as written.

## Install

```bash
pip install -e .              # data preparation and baselines
pip install -e ".[llm]"       # LoRA fine-tuning (CUDA GPU)
pip install -e ".[serve]"     # vLLM inference
git clone https://github.com/YangZhengyi98/DROS third_party/DROS && git -C third_party/DROS checkout 44ff99d
```

The official DROS code has no licence file, so it is not included here; `train_dros.py` imports it from
`third_party/DROS`.

## Run

```bash
# 1. Cohorts: 50,000 training-bank customers (40,000 training, 10,000 validation), 1,000 test-bank customers
python scripts/prepare_data.py \
    --train-transactions bank_a/transactions.csv --train-demographics bank_a/demographics.csv \
    --test-transactions bank_b/transactions.csv --test-demographics bank_b/demographics.csv

# 2. Baselines: trained at length 9, predicted at lengths 4, 7, 9 and 14
python scripts/run_baselines.py        # Random, Proportional, Averaging
python scripts/train_lstm.py
python scripts/train_cnn.py
python scripts/train_sasrec.py
python scripts/train_dros.py

# 3. LLMs: fine-tune and merge, then predict
python scripts/train_qwen.py
python scripts/predict_llm.py --family qwen --model runs/qwen-full/merged
python scripts/train_mistral.py
python scripts/predict_llm.py --family mistral --model runs/mistral-full/merged

# 4. Metrics and paired tests
python scripts/evaluate.py
python scripts/evaluate.py --fixed-population --out results/metrics_fixed_population
```

Outputs: `runs/` (checkpoints, adapters, merged models), `results/predictions/` (one CSV per model and
length with `customer_id`, `ground_truth`, `prediction`, `parsed_prediction`) and `results/metrics/`.
Baseline options that take several values (for example `--lr 1e-3 5e-4`) run a grid search on the
validation split.

## Ablations

| Condition | Command |
|---|---|
| Raw model, no fine-tuning | `predict_llm.py --family qwen --model Qwen/Qwen3.5-9B --name qwen-raw --lengths 9` |
| Category sequence only (Q-seq) | `train_qwen.py --condition seq`, then `predict_llm.py ... --condition seq` |
| 2^3 context factorial | `--condition seq+amounts`, `seq+dates`, `seq+demographics`, `seq+amounts+dates`, ... |
| Training seeds | `train_qwen.py --seed 2`, `--seed 3` (predict with `--name qwen-full-seed2`, ...) |
| Demographic leave-one-field-out | `train_qwen.py --drop-field gender` (or `age`, `marital_status`, `education`, `occupation`, `income_group`) |
| Customer identifier removed | `predict_llm.py --family qwen --model runs/qwen-full/merged --no-customer-id --lengths 9` |

## Settings

All values are in `src/mcc_llm/config.py`.

| | |
|---|---|
| LoRA | r 32, alpha 64, dropout 0.05; q, k, v, o, gate, up and down projections |
| Base weights | 8-bit (bitsandbytes), frozen |
| Optimisation | paged AdamW 8-bit, learning rate 2e-4, cosine decay, 100 warm-up steps, weight decay 0.01, gradient clipping 0.3 |
| Batch and epochs | 2 per device x 8 accumulation steps (16); 3 epochs, final checkpoint, no early stopping |
| Other | bf16, maximum length 512 tokens, seed 42 |
| Inference | adapter merged into bf16 weights, vLLM 0.30.0, greedy, at most 10 new tokens |
| Evaluation | 10,000 paired bootstrap resamples (seed 42), exact McNemar test, Holm over the five baselines |

The paper's LLM runs used one NVIDIA RTX 5090 (32 GB). The baselines train in minutes on a GPU and also
run on a CPU.

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

#!/usr/bin/env bash
# Run order of the whole pipeline. Run one block at a time; every block is safe to re-run.
set -euo pipefail

# --- 0. raw bank files -> input schema (on the machine that holds the bank data) ---------------------
#     One JSON spec per bank; see examples/export_spec.example.json. Specs and data stay out of the repository.
python scripts/export_raw.py --spec specs/bank_a.json --out raw/bank_a
python scripts/export_raw.py --spec specs/bank_b.json --out raw/bank_b

# --- 1. cohorts and splits (CPU, minutes) -------------------------------------------------------------
#     old/earlier_test_customers.csv: customer_id of every test customer of an earlier evaluation
python scripts/prepare_data.py \
  --train-transactions raw/bank_a/transactions.csv --train-demographics raw/bank_a/demographics.csv \
  --test-transactions  raw/bank_b/transactions.csv --test-demographics  raw/bank_b/demographics.csv \
  --dev-ids old/earlier_test_customers.csv --exclude old/earlier_test_customers.csv
python scripts/p0_checks.py cohort --data data/processed
python scripts/manifest.py data/processed --out results/MANIFEST_data
#     Fill in ANALYSIS_PLAN.md and commit it with results/MANIFEST_data BEFORE any training.
#     On the GPU machine, after copying data/processed:  python scripts/manifest.py data/processed --check results/MANIFEST_data

# --- 2. non-LLM models (CPU or GPU gaps; about 13 GPU-hours with the full grids) -----------------------
#     Each script trains, saves every model in runs/ and scores val, bank_a_test and bank_b_dev.
#     scripts/train_baselines.sh runs the six of them in this order, with logs, and can be restarted.
python scripts/run_baselines.py
python scripts/train_features.py
python scripts/train_lstm.py
python scripts/train_cnn.py
python scripts/train_sasrec.py
python scripts/train_dros.py                  # needs third_party/DROS at commit 44ff99d

# --- 3. GPU queue (priorities A, B, C, D in queue_5090.txt; resumable; skips finished runs) -----------
mkdir -p logs
nohup scripts/queue.sh queue_5090.txt > logs/queue_5090.out 2>&1 &
# follow it with: tail -f logs/queue_5090.out   and   tail -f logs/qwen-full.log

# --- 4. scoring that is allowed before the freeze ------------------------------------------------------
scripts/score_all.sh pre          # LLMs on val, bank_a_test and bank_b_dev (may run on a second GPU during training)
scripts/score_all.sh generation   # free generation on bank_b_dev: answers that are not a category (needs vLLM)

# --- 5. the freeze: every model is trained and selected; nothing has been scored on bank_b_confirm ---
python scripts/manifest.py runs --out results/MANIFEST_models      # adapters and saved models; no merged/ folders
git add results/MANIFEST_models && git commit -m "Freeze: manifest of the trained models" && git tag v3-prereg
git push origin main v3-prereg

# --- 6. confirmation scoring, once ---------------------------------------------------------------------
scripts/score_all.sh confirm      # every model on bank_b_confirm and bank_b_unfiltered
python scripts/manifest.py results/predictions --out results/MANIFEST_predictions

# --- 7. evaluation -------------------------------------------------------------------------------------
NON_LLM="averaging last-category markov1 lstm cnn sasrec dros-sasrec gbdt-seq gbdt-full logit-seq logit-full"
python scripts/evaluate_v3.py primary --llm qwen-full
python scripts/evaluate_v3.py primary --llm qwen-full --split bank_b_unfiltered      # sensitivity: no diversity filter
python scripts/evaluate_v3.py primary --llm qwen-full --without-ties                 # sensitivity: tied target dates
for split in bank_b_confirm bank_a_test; do
  python scripts/evaluate.py --predictions results/predictions/$split --out results/metrics/$split \
    --llms mistral-full qwen-seq qwen-full --baselines $NON_LLM \
    --pairs qwen-full:qwen-raw qwen-full:qwen-raw16 mistral-full:mistral-raw \
            qwen-full,qwen-full-seed2,qwen-full-seed3:qwen-seq,qwen-seq-seed2,qwen-seq-seed3 \
            qwen-seq,qwen-seq-seed2,qwen-seq-seed3:qwen-neutral,qwen-neutral-seed2,qwen-neutral-seed3
  python scripts/evaluate_v3.py summary --split $split
done
python scripts/evaluate.py --predictions results/predictions/bank_b_unfiltered --out results/metrics/bank_b_unfiltered \
  --lengths 9 --llms mistral-full qwen-seq qwen-full --baselines $NON_LLM
python scripts/evaluate.py --predictions results/predictions/bank_b_confirm --out results/metrics/bank_b_confirm_fixed_population \
  --fixed-population --pairs qwen-full@9:qwen-full@4 qwen-full@9:qwen-full@14 qwen-full-varlen@9:qwen-full-varlen@14
python scripts/evaluate_v3.py families --llms qwen-full qwen-seq mistral-full
python scripts/evaluate_v3.py shift
python scripts/evaluate_v3.py subgroups --models qwen-full mistral-full gbdt-full
python scripts/evaluate_v3.py calibration --model qwen-full --cost-fn 5 --cost-fp 1

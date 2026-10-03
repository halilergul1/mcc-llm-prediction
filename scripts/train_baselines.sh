#!/usr/bin/env bash
# Train every non-LLM model, one after the other, and score the splits that may be scored before the freeze
# (val, bank_a_test, bank_b_dev). Each step logs to logs/<step>.log.
#
#   mkdir -p logs && nohup scripts/train_baselines.sh > logs/train_baselines.out 2>&1 &
#
# The script can be started again after a crash: a finished step is skipped, and a step whose models are
# already saved only scores them again instead of retraining. The confirmation splits are not scored here;
# scripts/score_all.sh confirm does that after the freeze.
set -uo pipefail
[[ -f data/processed/train_transactions.csv ]] || { echo "data/processed is missing: copy the prepared splits here first"; exit 2; }
[[ -f third_party/DROS/SASRec_bce.py ]] || echo "warning: third_party/DROS is missing, so the dros-sasrec step will fail (README, Install)"
mkdir -p logs
failed=0

step() {   # step <name> <file that exists once the models are saved, or -> <script> [arguments...]
  local name=$1 trained=$2 script=$3; shift 3
  if [[ -f logs/$name.done ]]; then echo "$(date '+%F %T') skip  $name (done)"; return 0; fi
  local mode=()
  if [[ $trained != - && -f $trained ]]; then mode=(--score-only); echo "$(date '+%F %T') $name: models found, scoring only"; fi
  echo "$(date '+%F %T') start $name"
  local start; start=$(date +%s)
  if python "scripts/$script" ${mode[@]+"${mode[@]}"} "$@" >> "logs/$name.log" 2>&1 < /dev/null; then
    touch "logs/$name.done"
    echo "$(date '+%F %T') done  $name in $(( ($(date +%s) - start) / 60 )) min"
  else
    echo "$(date '+%F %T') FAIL  $name: see logs/$name.log; continuing with the next step"
    failed=1
  fi
}

step heuristics   -                               run_baselines.py      # Random, Proportional, Averaging, last-category, Markov-1
step features     runs/features/selection.json    train_features.py     # LightGBM and logistic regression (CPU)
step lstm         runs/lstm/training_log.json     train_lstm.py
step cnn          runs/cnn/training_log.json      train_cnn.py
step sasrec       runs/sasrec/training_log.json   train_sasrec.py
step dros-sasrec  runs/dros-sasrec/training_log.json train_dros.py      # needs third_party/DROS at commit 44ff99d

if [[ $failed == 0 ]]; then echo "$(date '+%F %T') all non-LLM models trained and scored"; else echo "$(date '+%F %T') finished with failures"; fi
exit $failed

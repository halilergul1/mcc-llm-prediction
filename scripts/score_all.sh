#!/usr/bin/env bash
# Score every trained model. LLMs: merge -> score -> delete the merged copy, one model at a time (disk stays low).
#
#   scripts/score_all.sh pre         LLMs on val, bank_a_test and bank_b_dev. Allowed BEFORE the freeze, e.g. on a
#                                    second GPU while the first is still training (none of these is the
#                                    confirmation data). The non-LLM models score every split when they are trained.
#   scripts/score_all.sh generation  free generation of the core LLMs on bank_b_dev at length 9: counts answers
#                                    that are not a category. Allowed before the freeze. It uses vLLM;
#                                    GEN_BACKEND=transformers scripts/score_all.sh generation works without it.
#   scripts/score_all.sh confirm     the LLMs on bank_b_confirm and bank_b_unfiltered. ONLY AFTER the freeze:
#                                    ANALYSIS_PLAN.md and results/MANIFEST_models committed and tagged v3-prereg.
#                                    A non-LLM model that is missing on these splits is scored from its saved models.
#
# Lengths: bank_b_confirm and bank_b_dev at 4, 7, 9 and 14; val, bank_a_test and bank_b_unfiltered at 9
# (what the analysis plan uses them for). A model already scored for a phase is skipped, so a crashed run
# can simply be restarted. Every step logs to logs/score_<name>.log.
set -euo pipefail
phase=${1:?usage: scripts/score_all.sh pre|generation|confirm}
[[ $phase == pre || $phase == generation || $phase == confirm ]] || { echo "unknown phase: $phase"; exit 2; }
mkdir -p logs
P=results/predictions
if [[ $phase == confirm ]]; then      # the LLMs are scored on the confirmation data only after the freeze (ANALYSIS_PLAN.md, section 6)
  if [[ ! -f results/MANIFEST_models ]] || ! git rev-parse -q --verify refs/tags/v3-prereg > /dev/null; then
    echo "the freeze comes first: commit results/MANIFEST_models and set the tag v3-prereg"; exit 1
  fi
fi

score() {   # score <family> <run-name> <condition> <all-lengths: yes|no> [extra predict_llm args...]
  local fam=$1 name=$2 cond=$3 all=$4; shift 4
  local first last marker lengths=()
  if [[ $phase == pre ]]; then first=bank_b_dev; last="val bank_a_test"; marker="$P/bank_a_test/${name}_last9.csv"
  else first=bank_b_confirm; last=bank_b_unfiltered; marker="$P/bank_b_unfiltered/${name}_last9.csv"; fi
  if [[ -f $marker ]]; then echo "skip $name ($phase done)"; return 0; fi
  [[ -d runs/$name/adapter ]] || { echo "missing runs/$name/adapter, skipped"; return 0; }
  [[ $all == yes ]] || lengths=(--lengths 9)
  echo "$(date '+%F %T') $phase $name"
  python scripts/merge.py --family "$fam" --run "runs/$name" >> "logs/score_$name.log" 2>&1
  local common=(--family "$fam" --model "runs/$name/merged" --name "$name" --condition "$cond" "$@")
  python scripts/predict_llm.py "${common[@]}" --splits $first ${lengths[@]+"${lengths[@]}"} >> "logs/score_$name.log" 2>&1
  python scripts/predict_llm.py "${common[@]}" --splits $last --lengths 9 >> "logs/score_$name.log" 2>&1
  rm -rf "runs/$name/merged"
}

generate() {   # generate <family> <run-name> <condition>
  local fam=$1 name=$2 cond=$3
  if [[ -f "$P/bank_b_dev/${name}-gen_last9.csv" ]]; then echo "skip $name (generation done)"; return 0; fi
  [[ -d runs/$name/adapter ]] || { echo "missing runs/$name/adapter, skipped"; return 0; }
  echo "$(date '+%F %T') generation $name"
  python scripts/merge.py --family "$fam" --run "runs/$name" >> "logs/score_$name.log" 2>&1
  python scripts/predict_llm.py --family "$fam" --model "runs/$name/merged" --name "$name" --condition "$cond" \
    --generate --backend "${GEN_BACKEND:-vllm}" --splits bank_b_dev --lengths 9 >> "logs/score_$name.log" 2>&1
  rm -rf "runs/$name/merged"
}

if [[ $phase == generation ]]; then
  generate qwen qwen-full full
  generate qwen qwen-seq seq
  generate qwen qwen-neutral neutral
  generate mistral mistral-full full
  echo "$(date '+%F %T') generation: done"
  exit 0
fi

# non-LLM models are scored on the confirmation splits when they are trained; one that is missing is scored now
if [[ $phase == confirm ]]; then
  confirm=(--splits bank_b_confirm bank_b_unfiltered)
  [[ -f $P/bank_b_unfiltered/markov1_last9.csv ]] || python scripts/run_baselines.py "${confirm[@]}" >> logs/score_baselines.log 2>&1
  [[ -f $P/bank_b_unfiltered/logit-full_last9.csv ]] || python scripts/train_features.py --score-only "${confirm[@]}" >> logs/score_baselines.log 2>&1
  for m in lstm cnn sasrec dros; do
    name=$m; [[ $m == dros ]] && name=dros-sasrec
    [[ -f $P/bank_b_unfiltered/${name}_last9.csv ]] || python scripts/train_$m.py --score-only "${confirm[@]}" >> logs/score_baselines.log 2>&1
  done
  echo "$(date '+%F %T') confirm: non-LLM models scored"
fi

# core LLMs (all lengths); --check on the first seed of each condition
score qwen qwen-full full yes --check 200
score qwen qwen-full-seed2 full yes
score qwen qwen-full-seed3 full yes
score qwen qwen-seq seq yes --check 200
score qwen qwen-seq-seed2 seq yes
score qwen qwen-seq-seed3 seq yes
score qwen qwen-neutral neutral yes --check 200
score qwen qwen-neutral-seed2 neutral yes
score qwen qwen-neutral-seed3 neutral yes
score mistral mistral-full full yes --check 200
score mistral mistral-full-seed2 full yes
score mistral mistral-full-seed3 full yes
score qwen qwen-full-varlen full yes
# factorial cells and leave-one-field-out (length 9 only)
for c in seq+amounts seq+dates seq+demographics seq+amounts+dates seq+amounts+demographics seq+dates+demographics; do
  score qwen "qwen-$c" "$c" no
done
for f in gender age marital_status education occupation income_group; do
  score qwen "qwen-full-minus-${f//_/-}" full no --drop-field "$f"
done

# raw models (no adapter), length 9. Zero-shot: exact scoring. 16-shot: prompts are about 16 times longer,
# so first-token scoring; predict_llm.py falls back to exact scoring if the --check line says it must.
for fam in qwen mistral; do
  base=$(python -c "from mcc_llm import config; print(config.BASE_MODELS['$fam'])")
  if [[ $phase == pre ]]; then splits=(val bank_b_dev); marker_split=bank_b_dev; else splits=(bank_b_confirm); marker_split=bank_b_confirm; fi
  for kind in raw raw16; do
    marker="$P/$marker_split/$fam-${kind}_last9.csv"
    [[ -f $marker ]] && { echo "skip $fam-$kind ($phase done)"; continue; }
    if [[ $kind == raw ]]; then extra=(--score sequence); else extra=(--score first --few-shot --check 50); fi
    echo "$(date '+%F %T') $phase $fam-$kind"
    python scripts/predict_llm.py --family $fam --model "$base" --name "$fam-$kind" "${extra[@]}" \
      --splits "${splits[@]}" --lengths 9 >> "logs/score_$fam-$kind.log" 2>&1
  done
done
echo "$(date '+%F %T') $phase: all scored"

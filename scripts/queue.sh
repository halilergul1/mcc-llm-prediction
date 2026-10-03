#!/usr/bin/env bash
# Run a GPU queue one job at a time, skipping jobs that already finished and logging each job to logs/.
#   mkdir -p logs && nohup scripts/queue.sh queue_5090.txt > logs/queue_5090.out 2>&1 &
# Each non-comment line: <run-dir> <command ...>. A job counts as finished when <run-dir>/adapter/adapter_config.json
# exists; a job that crashed resumes from its last checkpoint when the queue is started again.
set -uo pipefail
queue=${1:?usage: scripts/queue.sh <queue file>}
mkdir -p logs
while read -r dir cmd; do
  [[ -z "${dir}" || "${dir}" == \#* ]] && continue
  if [[ -f "${dir}/adapter/adapter_config.json" ]]; then echo "$(date '+%F %T') skip  ${dir} (done)"; continue; fi
  log="logs/$(basename "${dir}").log"
  echo "$(date '+%F %T') start ${dir}"
  start=$(date +%s)
  if bash -c "${cmd} --run-dir ${dir}" >> "${log}" 2>&1 < /dev/null; then
    echo "$(date '+%F %T') done  ${dir} in $(( ($(date +%s) - start) / 60 )) min"
  else
    echo "$(date '+%F %T') FAIL  ${dir}: see ${log}; continuing with the next job"
  fi
done < "${queue}"

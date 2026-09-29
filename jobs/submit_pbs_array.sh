#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pilot_half_filling.json}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

TASKS=$("${PYTHON_EXECUTABLE}" -m fk_transport manifest --config "${CONFIG}" | "${PYTHON_EXECUTABLE}" -c 'import json,sys; print(json.load(sys.stdin)["tasks"])')
if [[ "${TASKS}" -lt 1 ]]; then
  echo "Manifest contains no tasks" >&2
  exit 2
fi

JOBS=$(((TASKS + BATCH_SIZE - 1) / BATCH_SIZE))
echo "Spectral points: ${TASKS}; points per PBS job: ${BATCH_SIZE}; PBS jobs: ${JOBS}"
qsub -t "0-$((JOBS - 1))" \
  -v "CONFIG=${CONFIG},PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE},BATCH_SIZE=${BATCH_SIZE},TASK_COUNT=${TASKS}" \
  jobs/run_pbs_array.sh

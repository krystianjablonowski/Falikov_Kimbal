#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pilot_half_filling.json}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

TASKS=$("${PYTHON_EXECUTABLE}" -m fk_transport manifest --config "${CONFIG}" | "${PYTHON_EXECUTABLE}" -c 'import json,sys; print(json.load(sys.stdin)["tasks"])')
if [[ "${TASKS}" -lt 1 ]]; then
  echo "Manifest contains no tasks" >&2
  exit 2
fi

qsub -t "0-$((TASKS - 1))" -v "CONFIG=${CONFIG},PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}" jobs/run_pbs_array.sh

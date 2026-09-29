#!/usr/bin/env bash
#PBS -N fk_transport
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
unset PYTHONHOME
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG="${CONFIG:-configs/pilot_half_filling.json}"
ARRAY_INDEX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASK_COUNT="${TASK_COUNT:-}"

if [[ -z "${ARRAY_INDEX}" ]]; then
  echo "PBS array index is unavailable" >&2
  exit 2
fi

export PYTHONPATH="${PWD}/src"
if [[ -z "${TASK_COUNT}" ]]; then
  TASK_COUNT=$("${PYTHON_EXECUTABLE}" -m fk_transport manifest --config "${CONFIG}" | "${PYTHON_EXECUTABLE}" -c 'import json,sys; print(json.load(sys.stdin)["tasks"])')
fi

ARRAY_NUMBER=$((10#${ARRAY_INDEX}))
START_INDEX=$((ARRAY_NUMBER * BATCH_SIZE))
STOP_INDEX=$((START_INDEX + BATCH_SIZE))
if [[ "${STOP_INDEX}" -gt "${TASK_COUNT}" ]]; then
  STOP_INDEX="${TASK_COUNT}"
fi

echo "PBS array ${ARRAY_INDEX}: spectral points ${START_INDEX}..$((STOP_INDEX - 1))"
FAILED=0
for ((TASK_INDEX = START_INDEX; TASK_INDEX < STOP_INDEX; TASK_INDEX++)); do
  if ! "${PYTHON_EXECUTABLE}" -m fk_transport sweep --config "${CONFIG}" --index "${TASK_INDEX}"; then
    echo "Spectral point ${TASK_INDEX} failed" >&2
    FAILED=1
  fi
done
exit "${FAILED}"

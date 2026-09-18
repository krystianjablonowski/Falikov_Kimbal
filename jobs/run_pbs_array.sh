#!/usr/bin/env bash
#PBS -N fk_transport
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONHOME
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG="${CONFIG:-configs/pilot_half_filling.json}"
ARRAY_INDEX="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-}}"

if [[ -z "${ARRAY_INDEX}" ]]; then
  echo "PBS array index is unavailable" >&2
  exit 2
fi

export PYTHONPATH="${PWD}/src"
"${PYTHON_EXECUTABLE}" -m fk_transport sweep --config "${CONFIG}" --index "${ARRAY_INDEX}"

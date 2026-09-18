#!/usr/bin/env bash
#PBS -N fk_merge
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG="${CONFIG:-configs/pilot_half_filling.json}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_EXECUTABLE}" -m fk_transport status --config "${CONFIG}"
"${PYTHON_EXECUTABLE}" -m fk_transport merge --config "${CONFIG}"

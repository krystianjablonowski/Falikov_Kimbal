#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-${PWD}/.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-4}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

for config in \
  configs/stage9_localization_convergence_eta_5e-4_n20001_q96.json \
  configs/stage9_localization_convergence_eta_2p5e-4_n40001_q96.json \
  configs/stage9_localization_convergence_eta_1p25e-4_n80001_q96.json \
  configs/stage9_localization_convergence_eta_5e-4_n20001_q192.json; do
  [[ -f "${config}" ]] || {
    echo "Missing ${config}; run prepare-localization-convergence first" >&2
    exit 2
  }
  BATCH_SIZE="${BATCH_SIZE}" PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" \
    bash jobs/submit_pbs_array.sh "${config}"
done

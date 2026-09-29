#!/usr/bin/env bash
set -euo pipefail
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
for config in \
  configs/stage3_convergence_eta_5e-4.json \
  configs/stage3_convergence_eta_2p5e-4.json \
  configs/stage3_convergence_eta_1p25e-4.json; do
  [[ -f "${config}" ]] || { echo "Missing ${config}; run prepare-convergence first" >&2; exit 2; }
  BATCH_SIZE=1 PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}" bash jobs/submit_pbs_array.sh "${config}"
done

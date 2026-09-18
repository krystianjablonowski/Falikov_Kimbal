#!/usr/bin/env bash
#PBS -N fk_publish
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG="${CONFIG:-configs/pilot_half_filling.json}"
RUN_LABEL="${RUN_LABEL:-$(basename "${CONFIG}" .json)}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_EXECUTABLE}" -m fk_transport status --config "${CONFIG}"
"${PYTHON_EXECUTABLE}" -m fk_transport merge --config "${CONFIG}"

RESULT_ROOT=$("${PYTHON_EXECUTABLE}" -c 'import sys; from fk_transport.config import load_config; from fk_transport.sweep import output_root; print(output_root(load_config(sys.argv[1])))' "${CONFIG}")
DESTINATION="published_results/${RUN_LABEL}"
mkdir -p "${DESTINATION}"

for name in summary.csv status.json validation_report.json rerun_indices.txt; do
  if [[ -f "${RESULT_ROOT}/${name}" ]]; then
    cp "${RESULT_ROOT}/${name}" "${DESTINATION}/${name}"
  fi
done
cp "${CONFIG}" "${DESTINATION}/config.json"

git add "${DESTINATION}"
if git diff --cached --quiet; then
  echo "No changed published results to commit"
  exit 0
fi

git commit -m "results: ${RUN_LABEL}"
git push origin main

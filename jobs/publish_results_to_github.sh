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
RESULTS_BRANCH="${RESULTS_BRANCH:-results}"
PUBLISH_MODE="${PUBLISH_MODE:-summary}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_EXECUTABLE}" -m fk_transport status --config "${CONFIG}"
"${PYTHON_EXECUTABLE}" -m fk_transport merge --config "${CONFIG}"

RESULT_ROOT=$("${PYTHON_EXECUTABLE}" -c 'import sys; from fk_transport.config import load_config; from fk_transport.sweep import output_root; print(output_root(load_config(sys.argv[1])))' "${CONFIG}")
PUBLISH_WORKTREE="${PWD}/.publish-results-${PBS_JOBID:-$$}"

cleanup() {
  git worktree remove --force "${PUBLISH_WORKTREE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if git ls-remote --exit-code --heads origin "${RESULTS_BRANCH}" >/dev/null 2>&1; then
  git fetch origin "${RESULTS_BRANCH}"
  git worktree add -B "${RESULTS_BRANCH}" "${PUBLISH_WORKTREE}" "origin/${RESULTS_BRANCH}"
else
  git worktree add --detach "${PUBLISH_WORKTREE}" HEAD
  git -C "${PUBLISH_WORKTREE}" switch --orphan "${RESULTS_BRANCH}"
  git -C "${PUBLISH_WORKTREE}" rm -rf .
fi

DESTINATION="${PUBLISH_WORKTREE}/${RUN_LABEL}"
mkdir -p "${DESTINATION}"

if [[ "${PUBLISH_MODE}" == "full" ]]; then
  cp -a "${RESULT_ROOT}/." "${DESTINATION}/"
elif [[ "${PUBLISH_MODE}" == "summary" ]]; then
  for name in summary.csv status.json validation_report.json rerun_indices.txt manifest.json; do
    if [[ -f "${RESULT_ROOT}/${name}" ]]; then
      cp "${RESULT_ROOT}/${name}" "${DESTINATION}/${name}"
    fi
  done
else
  echo "PUBLISH_MODE must be 'summary' or 'full'" >&2
  exit 2
fi
cp "${CONFIG}" "${DESTINATION}/config.json"

if find "${DESTINATION}" -type f -size +95M -print -quit | grep -q .; then
  echo "A result file exceeds 95 MB. Use Git LFS or publish only summaries." >&2
  exit 3
fi

git -C "${PUBLISH_WORKTREE}" add "${RUN_LABEL}"
if git -C "${PUBLISH_WORKTREE}" diff --cached --quiet; then
  echo "No changed published results to commit"
  exit 0
fi

git -C "${PUBLISH_WORKTREE}" commit -m "results: ${RUN_LABEL}"
git -C "${PUBLISH_WORKTREE}" push origin "${RESULTS_BRANCH}"

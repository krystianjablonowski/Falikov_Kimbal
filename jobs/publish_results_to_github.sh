#!/usr/bin/env bash
#PBS -N fk_publish
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG="${CONFIG:-configs/pilot_half_filling.json}"
RUN_LABEL="${RUN_LABEL:-$(basename "${CONFIG}" .json)}"
RESULTS_BRANCH="${RESULTS_BRANCH:-results}"
PUBLISH_MODE="${PUBLISH_MODE:-summary}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_EXECUTABLE}" -m fk_transport status --config "${CONFIG}"
"${PYTHON_EXECUTABLE}" -m fk_transport merge --config "${CONFIG}"
"${PYTHON_EXECUTABLE}" -m fk_transport plot --config "${CONFIG}"

RESULT_ROOT=$("${PYTHON_EXECUTABLE}" -c 'import sys; from fk_transport.config import load_config; from fk_transport.sweep import output_root; print(output_root(load_config(sys.argv[1])))' "${CONFIG}")
PUBLISH_WORKTREE="${PWD}/.publish-results-${PBS_JOBID:-$$}"

cleanup() {
  git worktree remove --force "${PUBLISH_WORKTREE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git fetch origin "+refs/heads/${RESULTS_BRANCH}:refs/remotes/origin/${RESULTS_BRANCH}" || true
if git show-ref --verify --quiet "refs/remotes/origin/${RESULTS_BRANCH}"; then
  # A detached worktree avoids collisions with an existing local branch named
  # like the publication branch. The final push explicitly updates the remote.
  git worktree add --detach "${PUBLISH_WORKTREE}" "origin/${RESULTS_BRANCH}"
else
  git worktree add --detach "${PUBLISH_WORKTREE}" HEAD
  TEMPORARY_BRANCH="publish-${RESULTS_BRANCH}-${PBS_JOBID:-$$}"
  git -C "${PUBLISH_WORKTREE}" switch --orphan "${TEMPORARY_BRANCH}"
  git -C "${PUBLISH_WORKTREE}" rm -rf . || true
fi

DESTINATION="${PUBLISH_WORKTREE}/${RUN_LABEL}"
case "${DESTINATION}" in
  "${PUBLISH_WORKTREE}"/*) ;;
  *)
    echo "Refusing to refresh destination outside the publish worktree: ${DESTINATION}" >&2
    exit 4
    ;;
esac
rm -rf -- "${DESTINATION}"
mkdir -p "${DESTINATION}"

if [[ "${PUBLISH_MODE}" == "full" ]]; then
  cp -a "${RESULT_ROOT}/." "${DESTINATION}/"
elif [[ "${PUBLISH_MODE}" == "summary" ]]; then
  for source in \
    "${RESULT_ROOT}/summary.csv" \
    "${RESULT_ROOT}/activation_summary.csv" \
    "${RESULT_ROOT}/status.json" \
    "${RESULT_ROOT}/validation_report.json" \
    "${RESULT_ROOT}/rerun_indices.txt" \
    "${RESULT_ROOT}/manifest.json" \
    "${RESULT_ROOT}"/*_summary*.png \
    "${RESULT_ROOT}"/*_heatmap*.png \
    "${RESULT_ROOT}"/*boundary*.png \
    "${RESULT_ROOT}"/combined_summary.csv \
    "${RESULT_ROOT}"/relative_boundary_crossings.csv \
    "${RESULT_ROOT}"/relative_boundary_differences.csv \
    "${RESULT_ROOT}"/relative_boundary_separation_summary.csv \
    "${RESULT_ROOT}"/gradient_boundaries.csv \
    "${RESULT_ROOT}"/relative_boundary_level_status.csv \
    "${RESULT_ROOT}"/finite_filling_derived.csv \
    "${RESULT_ROOT}"/*.pdf; do
    if [[ -f "${source}" ]]; then
      cp "${source}" "${DESTINATION}/$(basename "${source}")"
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
git -C "${PUBLISH_WORKTREE}" push origin "HEAD:${RESULTS_BRANCH}"

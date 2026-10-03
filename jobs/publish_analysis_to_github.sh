#!/usr/bin/env bash
set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"
SOURCE_DIRECTORY="${SOURCE_DIRECTORY:?set SOURCE_DIRECTORY}"
RUN_LABEL="${RUN_LABEL:-stage3_convergence_analysis}"
RESULTS_BRANCH="${RESULTS_BRANCH:-results}"
PUBLISH_REPLACE="${PUBLISH_REPLACE:-0}"
[[ -d "${SOURCE_DIRECTORY}" ]] || { echo "Missing ${SOURCE_DIRECTORY}" >&2; exit 2; }
[[ "${PUBLISH_REPLACE}" == "0" || "${PUBLISH_REPLACE}" == "1" ]] || {
  echo "PUBLISH_REPLACE must be 0 (preserve remote files) or 1 (replace directory)" >&2
  exit 2
}
WORKTREE="${PWD}/.publish-analysis-${PBS_JOBID:-$$}"
cleanup(){ git worktree remove --force "${WORKTREE}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
git fetch origin "+refs/heads/${RESULTS_BRANCH}:refs/remotes/origin/${RESULTS_BRANCH}"
git worktree add --detach "${WORKTREE}" "origin/${RESULTS_BRANCH}"
DESTINATION="${WORKTREE}/${RUN_LABEL}"
case "${DESTINATION}" in "${WORKTREE}"/*) ;; *) exit 4;; esac
if [[ "${PUBLISH_REPLACE}" == "1" ]]; then
  rm -rf -- "${DESTINATION}"
fi
mkdir -p "${DESTINATION}"
for source in "${SOURCE_DIRECTORY}"/*.csv "${SOURCE_DIRECTORY}"/*.json "${SOURCE_DIRECTORY}"/*.pdf;do
  [[ -f "${source}" ]] && cp "${source}" "${DESTINATION}/"
done
git -C "${WORKTREE}" add "${RUN_LABEL}"
git -C "${WORKTREE}" diff --cached --quiet && { echo "No changed analysis results";exit 0; }
git -C "${WORKTREE}" commit -m "results: ${RUN_LABEL}"
git -C "${WORKTREE}" push origin "HEAD:${RESULTS_BRANCH}"

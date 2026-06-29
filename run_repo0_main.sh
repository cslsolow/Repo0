#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_NAME="${REPO_NAME:-requests}"
MODEL="${MODEL:-gpt-5-mini}"
BASE_URL="${BASE_URL:-https://api.qingyuntop.top/v1}"
API_KEY="${API_KEY:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
ACTION_REVISE_ROUNDS="${ACTION_REVISE_ROUNDS:-8}"

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: set API_KEY before running Repo0." >&2
  exit 2
fi

REQ_PATH="${REQ_PATH:-${ROOT_DIR}/repo_input/${REPO_NAME}/README.req}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${REPO_NAME}}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${OUTPUT_DIR}/readme_output/requirements.json}"

mkdir -p "${OUTPUT_DIR}"

cat <<EOF
Repo0 main artifact pipeline

1. Extract requirements from README.req / README-derived input.
2. Decompose requirements into subrequirements.
3. Generate the initial repository architecture.
4. Add missing actions when input requirements are not covered.
5. Compute component cohesion and coupling.
6. Use metrics to drive split and merge:
   - split: low-cohesion metric trigger plus requirement-graph partition evidence for LLM rewriting.
   - merge: metric candidate reviewed by the LLM judge.
   - repeat until no split or merge is proposed.
7. Run the final revise round over the optimized architecture.
8. Generate the complete repository from the final framework.

Repo: ${REPO_NAME}
Output: ${OUTPUT_DIR}
EOF

python "${ROOT_DIR}/run_agents.py" \
  --repo "${REPO_NAME}" \
  --workspace "${ROOT_DIR}" \
  --output "${OUTPUT_DIR}" \
  --req-path "${REQ_PATH}" \
  --requirements-file "${REQUIREMENTS_FILE}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --model "${MODEL}" \
  --max-workers "${MAX_WORKERS}" \
  --use-processes \
  --retry-empty-generated-components \
  --enable-gap-add-actions \
  --enable-component-metric-actions \
  --enable-component-metric-merge-judge \
  --action-refinement-rounds "${ACTION_REVISE_ROUNDS}" \
  --action-refinement-stop-on-stable \
  --parent-codegen-dag-source dependency

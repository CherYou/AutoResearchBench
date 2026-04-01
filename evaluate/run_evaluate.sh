#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${SCI_BENCH_ENV_FILE:-${REPO_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <deep|wide> [evaluation script args...]" >&2
    exit 1
fi

MODE="$1"
shift

case "${MODE}" in
    deep)
        SCRIPT_PATH="${SCRIPT_DIR}/evaluate_deep_search.py"
        ;;
    wide)
        SCRIPT_PATH="${SCRIPT_DIR}/evaluate_wide_search.py"
        ;;
    *)
        echo "Error: unsupported evaluation mode '${MODE}'. Use 'deep' or 'wide'." >&2
        exit 1
        ;;
esac

echo "Running ${MODE} evaluation..."
echo "Script: ${SCRIPT_PATH}"
echo "================================================================="

python3 "${SCRIPT_PATH}" "$@"

echo "================================================================="
echo "Evaluation finished successfully."

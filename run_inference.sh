#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCI_BENCH_ENV_FILE:-${SCRIPT_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
else
    echo "Error: .env file not found at ${ENV_FILE}" >&2
    exit 1
fi

INPUT_FILE="${INPUT_FILE:-input_data/academic_deepsearch_example.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-output_data/inference_output.jsonl}"
K_PASSES="${K_PASSES:-1}"
MAX_WORKERS="${MAX_WORKERS:-10}"
EVAL_START="${EVAL_START:-0}"
EVAL_END="${EVAL_END:-}"
MODEL="${MODEL:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_API_BASE="${OPENAI_API_BASE:-}"
MAX_TURNS="${MAX_TURNS:-10}"
SEARCH_TOOL="${SEARCH_TOOL:-deepxiv}"
VERBOSE="${VERBOSE:-0}"

if [[ -z "${MODEL}" ]]; then
    echo "Error: MODEL is required. Please set it in ${ENV_FILE}." >&2
    exit 1
fi

if [[ -z "${OPENAI_API_BASE}" ]]; then
    echo "Error: OPENAI_API_BASE is required. Please set it in ${ENV_FILE}." >&2
    exit 1
fi

CMD=(
    python3 "${SCRIPT_DIR}/inference.py"
    --input-file "${INPUT_FILE}"
    --output-file "${OUTPUT_FILE}"
    -k "${K_PASSES}"
    --max-workers "${MAX_WORKERS}"
    --eval-start "${EVAL_START}"
    --model "${MODEL}"
    --api-key "${OPENAI_API_KEY}"
    --api-base "${OPENAI_API_BASE}"
    --max-turns "${MAX_TURNS}"
    --search-tool "${SEARCH_TOOL}"
)

if [[ -n "${EVAL_END}" ]]; then
    CMD+=(--eval-end "${EVAL_END}")
fi

if [[ "${VERBOSE}" == "1" ]]; then
    CMD+=(--verbose)
fi

CMD+=("$@")

echo "Starting batch inference..."
echo "Input file:   ${INPUT_FILE}"
echo "Output file:  ${OUTPUT_FILE}"
echo "K value:      ${K_PASSES}"
echo "Max workers:  ${MAX_WORKERS}"
echo "Eval range:   ${EVAL_START} -> ${EVAL_END:-EOF}"
echo "Model:        ${MODEL}"
echo "Max turns:    ${MAX_TURNS}"
echo "Search tool:  ${SEARCH_TOOL}"
echo "================================================================="

"${CMD[@]}"

echo "================================================================="
echo "Inference finished successfully."
echo "Results have been saved to ${OUTPUT_FILE}"

#!/bin/bash
# Run vLLM inference for all 24 trained models sequentially.
#
# Output dirs are resolved from configs/paths.yaml (same source as training configs).
#
# Usage:
#   bash vllm_inference/scripts/run_inference_all.sh [--partition development|evaluation]
#
# To run a subset: comment out lines in the JOBS list below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATHS="${PATHS:-${REPO_ROOT}/configs/paths.yaml}"
PARTITION="evaluation"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve paths via paths_config.py (single source of truth).
read -r OUTPUT_DIR DATA_DIR OUTPUT_ROOT < <(python3 - <<EOF
import sys
sys.path.insert(0, "${REPO_ROOT}")
from paths_config import load_paths
p = load_paths("${PATHS}")
print(p.output_dir, p.data_dir, p.results_dir)
EOF
)

echo "output_dir : ${OUTPUT_DIR}"
echo "data_dir   : ${DATA_DIR}"
echo "results    : ${OUTPUT_ROOT}"
echo "partition  : ${PARTITION}"
echo ""

# ---------------------------------------------------------------------------
# Job list — one dataset name per line (same as run_name and output subdir).
# Comment out any you want to skip.
# ---------------------------------------------------------------------------
JOBS=(
    # 64k
    clinc150_64k
    hotpot_qa_64k
    infinite_bench_mc_64k
    infinite_bench_qa_64k
    json_kv_64k
    ms_macro_64k
    nlu_64k
    nq_64k
    pop_qa_64k
    ruler_mk_uuid_64k
    trec_coarse_64k
    trivia_qa_64k
    # 128k
    clinc150_128k
    hotpot_qa_128k
    infinite_bench_mc_128k
    infinite_bench_qa_128k
    json_kv_128k
    ms_macro_128k
    nlu_128k
    nq_128k
    pop_qa_128k
    ruler_mk_uuid_128k
    trec_coarse_128k
    trivia_qa_128k
)

FAILED=()

for DATASET in "${JOBS[@]}"; do
    # Find the best checkpoint; skip if training is not done yet.
    MODEL_PATH=$(echo "${OUTPUT_DIR}/${DATASET}"/v0-*/best 2>/dev/null | tr ' ' '\n' | head -1)
    if [[ ! -d "${MODEL_PATH:-}" ]]; then
        echo "SKIP  ${DATASET}: no checkpoint found (training incomplete or not started)"
        continue
    fi

    # Skip if results already exist (inference.py uses .tmp → rename, so a .jsonl
    # file is only written on success — no risk of treating partial results as done).
    RESULT_FILE="${OUTPUT_ROOT}/finetuned/${DATASET}/${PARTITION}.jsonl"
    if [[ -f "${RESULT_FILE}" ]]; then
        echo "SKIP  ${DATASET}: results already exist at ${RESULT_FILE}"
        continue
    fi

    echo ""
    echo "--- Inference: ${DATASET} ---"
    echo "Model: ${MODEL_PATH}"
    echo "Time:  $(date)"

    python "${REPO_ROOT}/vllm_inference/inference.py" \
        --model-path "${MODEL_PATH}" \
        --dataset    "${DATASET}" \
        --partition  "${PARTITION}" \
        --data-dir   "${DATA_DIR}" \
        --output-dir "${OUTPUT_ROOT}" \
        && echo "--- Done: ${DATASET} at $(date) ---" \
        || { echo "--- FAILED: ${DATASET} at $(date) ---" >&2; FAILED+=("${DATASET}"); }
done

echo ""
echo "========================================"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED jobs:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    exit 1
else
    echo "All inference jobs completed."
fi

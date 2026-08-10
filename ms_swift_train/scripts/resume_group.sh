#!/bin/bash
# Resume incomplete training jobs for a group of datasets.
#
# For each config in the group, finds the latest v0-* run directory under the
# dataset's output path, checks how many epochs completed vs. num_train_epochs
# in the config, and resumes if incomplete.
#
# Usage:
#   bash ms_swift_train/scripts/resume_group.sh <group_id>
#
# group_id must be one of: ec2_1_gpu0, ec2_1_gpu4, ec2_2_gpu0, ec2_2_gpu4, ec2_3_gpu0, ec2_3_gpu4
#
# Run both groups on the same instance in parallel:
#   bash ms_swift_train/scripts/resume_group.sh ec2_1_gpu0 &
#   bash ms_swift_train/scripts/resume_group.sh ec2_1_gpu4 &
#   wait

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/../configs"

source "${SCRIPT_DIR}/groups.sh"

GROUP="${1:-}"
if [[ -z "${GROUP}" ]]; then
    echo "Usage: $0 <group_id>"
    echo "Valid group IDs: ec2_1_gpu0 ec2_1_gpu4 ec2_2_gpu0 ec2_2_gpu4 ec2_3_gpu0 ec2_3_gpu4"
    exit 1
fi

resolve_group "${GROUP}" || exit 1

FIRST_GPU="${GPUS%%,*}"
PORT=$((29500 + FIRST_GPU))

NPROC="${NPROC:-4}"
PATHS="${PATHS:-${REPO_ROOT}/configs/paths.yaml}"

# Resolve output_dir base from paths.yaml
OUTPUT_BASE="$(python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from paths_config import load_paths
print(load_paths('${PATHS}').output_dir)
")"

echo "========================================"
echo "Group:       ${GROUP}"
echo "GPUs:        ${GPUS}"
echo "Port:        ${PORT}"
echo "Output base: ${OUTPUT_BASE}"
echo "Jobs:        ${#CONFIGS[@]}"
echo "Start:       $(date)"
echo "========================================"

FAILED=()
SKIPPED=()

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIGS_DIR}/${CONFIG}.yaml"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config not found: ${CONFIG_FILE}" >&2
        FAILED+=("${CONFIG}")
        continue
    fi

    # Derive dataset output dir from config path: "64k/clinc150" -> clinc150_64k
    CONTEXT="${CONFIG%%/*}"   # "64k"
    DATASET="${CONFIG##*/}"   # "clinc150"
    DATASET_DIR="${OUTPUT_BASE}/${DATASET}_${CONTEXT}"

    # Read target epochs from the config
    TARGET_EPOCHS="$(python3 -c "
import sys, yaml
sys.path.insert(0, '${REPO_ROOT}')
sys.path.insert(0, '${SCRIPT_DIR}/..')
from run_sft import load_config
c = load_config('${CONFIG_FILE}', '${PATHS}')
print(c.training.num_train_epochs)
")"

    # Find the latest v0-* run directory
    RUN_DIR="$(ls -dt "${DATASET_DIR}"/v0-* 2>/dev/null | head -1 || true)"

    if [[ -z "${RUN_DIR}" ]]; then
        echo ""
        echo "--- SKIP (no run dir): ${CONFIG} ---"
        echo "    Expected: ${DATASET_DIR}/v0-*"
        SKIPPED+=("${CONFIG}")
        continue
    fi

    LOGGING_JSONL="${RUN_DIR}/logging.jsonl"
    if [[ ! -f "${LOGGING_JSONL}" ]]; then
        echo ""
        echo "--- SKIP (no logging.jsonl): ${CONFIG} -> ${RUN_DIR} ---"
        SKIPPED+=("${CONFIG}")
        continue
    fi

    COMPLETED_EPOCHS="$(python3 -c "
import json
last = 0.0
with open('${LOGGING_JSONL}') as f:
    for line in f:
        try:
            e = json.loads(line)
            if 'epoch' in e:
                last = max(last, float(e['epoch']))
        except Exception:
            pass
print(last)
")"

    if python3 -c "exit(0 if float('${COMPLETED_EPOCHS}') < float('${TARGET_EPOCHS}') else 1)"; then
        echo ""
        echo "--- Resuming: ${CONFIG} (${COMPLETED_EPOCHS}/${TARGET_EPOCHS} epochs done) ---"
        echo "    Run dir: ${RUN_DIR}"
        echo "    Time: $(date)"

        CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
            --nproc_per_node="${NPROC}" \
            --master_port="${PORT}" \
            "${REPO_ROOT}/ms_swift_train/resume_training.py" \
            --run-dir "${RUN_DIR}" \
            --config "${CONFIG_FILE}" \
            --paths "${PATHS}" \
            && echo "--- Done: ${CONFIG} at $(date) ---" \
            || { echo "--- FAILED: ${CONFIG} at $(date) ---" >&2; FAILED+=("${CONFIG}"); }

        sleep 5
    else
        echo ""
        echo "--- SKIP (already complete): ${CONFIG} (${COMPLETED_EPOCHS}/${TARGET_EPOCHS} epochs) ---"
        SKIPPED+=("${CONFIG}")
    fi
done

echo ""
echo "========================================"
echo "Group ${GROUP} finished at $(date)"

if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "Skipped (complete or no run found):"
    for s in "${SKIPPED[@]}"; do echo "  $s"; done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED jobs:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    exit 1
else
    echo "All attempted jobs succeeded."
fi

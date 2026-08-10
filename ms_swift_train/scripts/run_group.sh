#!/bin/bash
# Run a group of training jobs sequentially on a specific set of GPUs.
#
# Usage:
#   bash ms_swift_train/scripts/run_group.sh <group_id>
#
# group_id must be one of: ec2_1_gpu0, ec2_1_gpu4, ec2_2_gpu0, ec2_2_gpu4, ec2_3_gpu0, ec2_3_gpu4
#
# Run both groups on the same instance in parallel:
#   bash ms_swift_train/scripts/run_group.sh ec2_1_gpu0 &
#   bash ms_swift_train/scripts/run_group.sh ec2_1_gpu4 &
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

echo "========================================"
echo "Group:   ${GROUP}"
echo "GPUs:    ${GPUS}"
echo "Port:    ${PORT}"
echo "Jobs:    ${#CONFIGS[@]}"
echo "Start:   $(date)"
echo "========================================"

FAILED=()

for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIGS_DIR}/${CONFIG}.yaml"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config not found: ${CONFIG_FILE}" >&2
        FAILED+=("${CONFIG}")
        continue
    fi

    echo ""
    echo "--- Starting: ${CONFIG} ---"
    echo "Time: $(date)"

    CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
        --nproc_per_node="${NPROC}" \
        --master_port="${PORT}" \
        "${REPO_ROOT}/ms_swift_train/run_sft.py" \
        --config "${CONFIG_FILE}" \
        --paths "${PATHS}" \
        && echo "--- Done: ${CONFIG} at $(date) ---" \
        || { echo "--- FAILED: ${CONFIG} at $(date) ---" >&2; FAILED+=("${CONFIG}"); }

    sleep 5
done

echo ""
echo "========================================"
echo "Group ${GROUP} finished at $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "FAILED jobs:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    exit 1
else
    echo "All jobs succeeded."
fi

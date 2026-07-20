#!/bin/bash
# Run a group of training jobs sequentially on a specific set of GPUs.
#
# Usage:
#   bash ms_swift_train/scripts/run_group.sh <group_id>
#
# group_id must be one of: ec2_1_gpu0, ec2_1_gpu4, ec2_2_gpu0, ec2_2_gpu4, ec2_3_gpu0, ec2_3_gpu4
#
# Each group corresponds to 4 GPUs on one EC2 instance. Jobs run one at a time;
# the next job starts only after the previous one finishes.
#
# Run both groups on the same instance in parallel:
#   bash ms_swift_train/scripts/run_group.sh ec2_1_gpu0 &
#   bash ms_swift_train/scripts/run_group.sh ec2_1_gpu4 &
#   wait

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/../configs"

GROUP="${1:-}"
if [[ -z "${GROUP}" ]]; then
    echo "Usage: $0 <group_id>"
    echo "Valid group IDs: ec2_1_gpu0 ec2_1_gpu4 ec2_2_gpu0 ec2_2_gpu4 ec2_3_gpu0 ec2_3_gpu4"
    exit 1
fi

# ---------------------------------------------------------------------------
# Group definitions — edit GPUS if your device IDs differ
# ---------------------------------------------------------------------------
case "${GROUP}" in
    ec2_1_gpu0)
        GPUS="0,1,2,3"
        CONFIGS=(
            "128k/clinc150"
            "64k/pop_qa"
            "64k/infinite_bench_mc"
        )
        ;;
    ec2_1_gpu4)
        GPUS="4,5,6,7"
        CONFIGS=(
            "128k/nlu"
            "64k/trivia_qa"
            "64k/json_kv"
            "64k/ms_macro"
        )
        ;;
    ec2_2_gpu0)
        GPUS="0,1,2,3"
        CONFIGS=(
            "128k/nq"
            "64k/nq"
            "64k/hotpot_qa"
            "128k/ms_macro"
        )
        ;;
    ec2_2_gpu4)
        GPUS="4,5,6,7"
        CONFIGS=(
            "128k/trivia_qa"
            "128k/pop_qa"
            "128k/infinite_bench_qa"
            "128k/json_kv"
            "128k/ruler_mk_uuid"
        )
        ;;
    ec2_3_gpu0)
        GPUS="0,1,2,3"
        CONFIGS=(
            "128k/trec_coarse"
            "64k/nlu"
            "64k/trec_coarse"
        )
        ;;
    ec2_3_gpu4)
        GPUS="4,5,6,7"
        CONFIGS=(
            "128k/hotpot_qa"
            "64k/clinc150"
            "128k/infinite_bench_mc"
            "64k/infinite_bench_qa"
            "64k/ruler_mk_uuid"
        )
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Valid group IDs: ec2_1_gpu0 ec2_1_gpu4 ec2_2_gpu0 ec2_2_gpu4 ec2_3_gpu0 ec2_3_gpu4"
        exit 1
        ;;
esac

# Use a stable port derived from the first GPU index to avoid collisions
# when both groups run in parallel on the same instance.
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
    JOB_NAME="$(echo "${CONFIG}" | tr '/' '_')"

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
        "${SCRIPT_DIR}/../../ms_swift_train/run_sft.py" \
        --config "${CONFIG_FILE}" \
        --paths "${PATHS}" \
        && echo "--- Done: ${CONFIG} at $(date) ---" \
        || { echo "--- FAILED: ${CONFIG} at $(date) ---" >&2; FAILED+=("${CONFIG}"); }
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

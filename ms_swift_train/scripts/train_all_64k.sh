#!/bin/bash
# Run all 64k training configs sequentially.
# Set NPROC to the number of GPUs on your machine (default: 4).
set -euo pipefail

NPROC=${NPROC:-4}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${REPO_ROOT}/ms_swift_train/configs/64k"

for cfg in "${CONFIG_DIR}"/*.yaml; do
    echo "========================================"
    echo "Config: ${cfg}"
    echo "Start:  $(date)"
    echo "========================================"
    torchrun --nproc_per_node="${NPROC}" \
        "${REPO_ROOT}/ms_swift_train/run_sft.py" \
        --config "${cfg}" \
        --paths "${REPO_ROOT}/configs/paths.yaml"
    echo "Done: $(date)"
done

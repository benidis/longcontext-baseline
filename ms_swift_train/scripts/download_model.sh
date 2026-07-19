#!/bin/bash
# Download Qwen3-4B-Instruct-2507 to scratch using swift's HF hub backend.
# Run this from a login node (internet access required).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
PATHS_CONFIG="${REPO_ROOT}/configs/paths.yaml"

MODEL_DIR="$("${PYTHON}" -c "import sys; sys.path.insert(0, '${REPO_ROOT}'); from paths_config import load_paths; print(load_paths('${PATHS_CONFIG}').model_dir)")"
DEST="${MODEL_DIR}/Qwen/Qwen3-4B-Instruct-2507"

mkdir -p "$(dirname "${DEST}")"

USE_HF=1 "${PYTHON}" - <<EOF
from swift.utils.hub_utils import safe_snapshot_download
model_dir = safe_snapshot_download(
    "Qwen/Qwen3-4B-Instruct-2507",
    use_hf=True,
    local_dir="${DEST}",
)
print(f"Model downloaded to: {model_dir}")
EOF

#!/bin/bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -o pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load 2024 && module load CUDA/12.6.0
export CUDA_HOME=/sw/arch/RHEL9/EB_production/2024/software/CUDA/12.6.0


# SLURM batch shells are non-interactive and may not load your rc files.
if [[ -f "$HOME/.zshrc" ]]; then
	# shellcheck disable=SC1090
	source "$HOME/.zshrc"
fi
if [[ -f "$HOME/.bashrc" ]]; then
	# shellcheck disable=SC1090
	source "$HOME/.bashrc"
fi

# ---------------------------------------------------------------------------
# Configuration — edit this section to customise your run
# (placeholders: replace with your own locations)
# ---------------------------------------------------------------------------
PYTHON_BIN=/path/to/conda_env/bin/python
PROJECT_DIR=/path/to/vllm_inference_project

MODEL_ID_OR_PATH='Qwen/Qwen3-4B-Instruct-2507'

# Data and output paths
DATA_ROOT=/path/to/base_dir/helmet/longtrain_swift
OUT_ROOT=/path/to/results

# ---------------------------------------------------------------------------
# Datasets to evaluate — add or remove dataset names here, one per line
# ---------------------------------------------------------------------------
DATASETS=(
	nq_64k
	trivia_qa_64k
	pop_qa_64k
	hotpot_qa_64k
)

# ---------------------------------------------------------------------------
# You can also pass datasets as command-line arguments to override the list
# above, e.g.:  sbatch run_inference.sh nq_64k triviaqa popqa
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
	DATASETS=("$@")
fi

mkdir -p logs "${OUT_ROOT}"

echo "Job ID:    ${SLURM_JOB_ID:-<none>}"
echo "Node:      ${SLURMD_NODENAME:-<none>}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "Start:     $(date)"

cd "${PROJECT_DIR}"

for DATASET_NAME in "${DATASETS[@]}"; do
	EVAL_DATASET="${DATA_ROOT}/${DATASET_NAME}/evaluation.jsonl"

	echo "========================================"
	echo "Dataset: ${DATASET_NAME}"
	echo "Eval source: ${EVAL_DATASET}"
	echo "========================================"

	if ! "${PYTHON_BIN}" inference.py \
		--model-path "${MODEL_ID_OR_PATH}" \
		--dataset "${DATASET_NAME}" \
		--partition evaluation \
		--data-dir "${DATA_ROOT}" \
		--output-dir "${OUT_ROOT}" \
		--concurrency 16; then
		echo "WARNING: Inference failed for ${DATASET_NAME}." >&2
	fi
done

echo "End: $(date)"

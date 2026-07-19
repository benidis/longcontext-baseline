# Long-Context Baseline

LoRA fine-tuning baseline for long-context experiments on 12 HELMET datasets,
using [ms-swift](https://github.com/modelscope/ms-swift) as the training backend.

> **Validate before running the full sweep.** Pick one dataset (e.g. `clinc150`)
> at 64k, run the full pipeline end-to-end (data → train → inference → evaluate),
> then launch the rest. Full 64k runs take on the order of 5–10 hours per dataset
> pair; 128k runs take considerably longer.

## Repository Structure

```
.
├── pyproject.toml
├── configs/
│   └── paths.yaml                          # machine-local paths (edit once per machine)
├── create_data.py                          # Step 1: download & prepare the 12 HELMET datasets
├── ms_swift_train/
│   ├── convert_to_swift_messages.py        # Step 2: convert datasets to ms-swift JSONL
│   ├── run_sft.py                          # Step 3: training entry point
│   ├── configs/
│   │   ├── base.yaml                       # shared hyperparameters (LoRA, LR, hardware, …)
│   │   ├── 64k/                            # per-run deltas for 64k context
│   │   │   ├── clinc150_ruler.yaml
│   │   │   ├── hotpot_popqa.yaml
│   │   │   ├── nlu_msmacro.yaml
│   │   │   ├── nq_jsonkv.yaml
│   │   │   ├── trec_infbenchmc.yaml
│   │   │   └── trivia_infbenchqa.yaml
│   │   └── 128k/                           # same 6 configs at 128k context
│   │       └── ...
│   └── scripts/
│       ├── download_model.sh               # download Qwen3-4B-Instruct-2507
│       ├── train_all_64k.sh                # run all 64k configs sequentially
│       └── train_all_128k.sh               # run all 128k configs sequentially
└── vllm_inference/
    ├── inference.py                        # Step 4: vLLM inference (base or LoRA adapter)
    ├── utils.py
    ├── evaluate.py                         # Step 5: SubEM scoring
    └── scripts/
        ├── run_inference.sh
        └── run_finetuned_inference.sh
```

Useful references:
- ms-swift sequence parallel examples: <https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel>
- HELMET datasets (`keys_values`): <https://github.com/awslabs/keys_values/blob/main/keys_values/data/helmet.md>

## Environment Setup

All experiments run on a machine with 4+ A100 GPUs.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a virtual environment

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Install the package

`pyproject.toml` defines the following extras:

- (no extra) — core deps only: `pydantic`, `pyyaml`. Enough for config loading.
- `[train]` — adds `deepspeed`, `liger-kernel`, `tensorboard`.
- `[data]` — adds `keys_values` for dataset preparation (`create_data.py`).
- `[inference]` — adds `vllm==0.23.0` for inference.
- `[dev]` — adds `pytest`, `black`, `isort`.

Install what you need:

```bash
uv pip install -e /home/ubuntu/Repos/keys_values
uv pip install -e ".[train,data,dev]"
```

### 4. Install ms-swift manually

ms-swift is not declared as a `pyproject.toml` dependency because it pulls in its
own torch version. Install it after the package, then pin torch to the correct
CUDA version afterwards.

```bash
uv pip install ms-swift -U
uv pip install 'ms-swift[megatron]' -U
```

### 5. Pin PyTorch to CUDA 12.8

ms-swift installs its own torch which may not match your system CUDA. Reinstall
the correct version after ms-swift:

```bash
uv pip install "torch==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
```

Verify it stuck:

```bash
python -c "import torch; print(torch.__version__)"  # should print 2.7.0+cu128
```

### 6. Install FlashAttention

FlashAttention requires `nvcc` at build time. On a standard GPU EC2 (DLAMI or
similar) `nvcc` is already available — verify with `nvcc --version`. If it is
missing, install it first:

```bash
sudo apt install -y nvidia-cuda-toolkit
```

Then install FlashAttention:

```bash
uv pip install flash-attn --no-build-isolation
```

### 7. Configure paths

All paths live in one file: [configs/paths.yaml](configs/paths.yaml).
Edit it once per machine before running anything:

```yaml
base_dir: /opt/dlami/nvme
```

Everything else is derived from `base_dir`:

| Path | Value |
|---|---|
| `model_dir` | `base_dir/checkpoints` |
| `output_dir` | `base_dir/output` |
| `data_dir` | `base_dir/helmet/longtrain_swift` |

Every script accepts `--paths configs/paths.yaml` to read from this file.

## Training Workflow

### Step 1: Download and prepare datasets

```bash
python create_data.py --paths configs/paths.yaml --max-length 64k
python create_data.py --paths configs/paths.yaml --max-length 128k
```

Output lands under `base_dir/helmet/longtrain/`.

### Step 2: Convert to ms-swift JSONL format

```bash
python ms_swift_train/convert_to_swift_messages.py --paths configs/paths.yaml
```

To convert a single dataset only:

```bash
python ms_swift_train/convert_to_swift_messages.py --paths configs/paths.yaml --dataset nq_64k
```

### Step 3: Download the base model

```bash
bash ms_swift_train/scripts/download_model.sh
```

The destination is read from `model_dir` in `configs/paths.yaml`.

### Step 4: Train

#### Option A — HuggingFace backend (default)

Uses the HuggingFace Trainer + DeepSpeed Ulysses sequence parallelism.

**Single run:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ms_swift_train/run_sft.py \
    --config ms_swift_train/configs/64k/clinc150_ruler.yaml \
    --paths configs/paths.yaml
```

**All 64k runs sequentially:**

```bash
bash ms_swift_train/scripts/train_all_64k.sh
```

**All 128k runs sequentially:**

```bash
bash ms_swift_train/scripts/train_all_128k.sh
```

Set `NPROC` to match your GPU count (default: 4):

```bash
NPROC=8 bash ms_swift_train/scripts/train_all_64k.sh
```

#### Option B — Megatron backend

Uses Megatron-Core instead of HuggingFace Trainer. Key differences from Option A:

- Sequence parallelism is **ring attention** (`context_parallel_size`), not Ulysses
- No DeepSpeed — uses Megatron's own DistributedOptimizer (behaves like ZeRO2)
- Batch size is specified as `micro_batch_size` (per GPU per step) and `global_batch_size` (total across all GPUs and gradient accumulation steps)
- Liger cross-entropy incompatibility does **not** apply here

Parameters can be passed as CLI flags or as a flat JSON/YAML config file
(`HfArgumentParser` detects a `.json` or `.yaml` argument and loads it):

```bash
# CLI flags
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    -m swift.cli._megatron.sft \
    --model_id_or_path ${MODEL_DIR}/Qwen/Qwen3-4B-Instruct-2507 \
    --dataset /path/to/dataset.jsonl \
    --output_dir ${OUTPUT_DIR} \
    --context_parallel_size 4 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --max_length 65536 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --learning_rate 5e-4 \
    --num_train_epochs 5

# Config file (flat key-value, all the same fields)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501\
    -m swift.cli._megatron.sft \
    ms_swift_train/configs/megatron/clinc150_ruler_64k.json
```

Example `clinc150_ruler_64k.json`:
```json
{
  "model_id_or_path": "/opt/dlami/nvme/longcontext/checkpoints/Qwen/Qwen3-4B-Instruct-2507",
  "dataset": ["/opt/dlami/nvme/longcontext/helmet/longtrain_swift/clinc150_64k/train.jsonl",
              "/opt/dlami/nvme/longcontext/helmet/longtrain_swift/ruler_mk_uuid_64k/train.jsonl"],
  "output_dir": "/opt/dlami/nvme/longcontext/output/clinc150_ruler_64k_megatron",
  "context_parallel_size": 4,
  "micro_batch_size": 1,
  "global_batch_size": 32,
  "max_length": 65536,
  "lora_rank": 16,
  "lora_alpha": 16,
  "learning_rate": 5e-4,
  "num_train_epochs": 5
}
```

Note: unlike `base.yaml`, this is a **flat** dict — no nesting, no `${var}` substitution,
no `base:` inheritance. All fields must be spelled out in full.

`dp_world_size` is derived automatically: `total_gpus / context_parallel_size`.
With 4 GPUs and `context_parallel_size=4`, dp=1 — same trade-off as the HF backend.
With `context_parallel_size=2`, dp=2 and Megatron's optimizer shards across 2 replicas.

> **Note:** the Megatron backend requires `ms-swift[megatron]` to be installed
> (see Environment Setup step 4).

### Config structure

`base.yaml` holds all shared hyperparameters (LoRA settings, learning rate,
hardware, logging, etc.). Each per-run config in `64k/` or `128k/` contains
only the delta — `max_length`, `data.datasets`, `output_dir`, and `run_name`.
Fields in the per-run config override the base.

> **Note:** the Megatron backend does not use `base.yaml` — all parameters must
> be passed as CLI flags or a separate config file in Megatron's own format.

## Inference

> The inference code targets **vLLM 0.23.0** — install exactly that version:
> ```bash
> uv pip install vllm==0.23.0
> ```

```bash
# Base model
python vllm_inference/inference.py \
    --model-path /opt/dlami/nvme/checkpoints/Qwen/Qwen3-4B-Instruct-2507 \
    --dataset nq_64k \
    --data-dir /opt/dlami/nvme/helmet/longtrain_swift \
    --output-dir /opt/dlami/nvme/results

# Fine-tuned (LoRA adapter)
python vllm_inference/inference.py \
    --model-path /opt/dlami/nvme/output/nq_jsonkv_64k \
    --dataset nq_64k \
    --data-dir /opt/dlami/nvme/helmet/longtrain_swift \
    --output-dir /opt/dlami/nvme/results
```

## Evaluation

```bash
python vllm_inference/evaluate.py /path/to/results
```

Scores results using **SubEM** (substring exact match).

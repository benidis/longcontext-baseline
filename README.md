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
│   │   ├── 64k/                            # one config per dataset at 64k context (12 files)
│   │   └── 128k/                           # one config per dataset at 128k context (12 files)
│   └── scripts/
│       └── download_model.sh               # download Qwen3-4B-Instruct-2507
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
- `[dev]` — adds `pytest`, `black`, `isort`.

Install what you need for training:

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

Then install FlashAttention. Pin to exactly `2.8.3` — `2.8.3.post1` has a packaging
quirk that breaks version detection in some backends:

```bash
uv pip install flash-attn==2.8.3 --no-build-isolation
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

Uses the HuggingFace Trainer + DeepSpeed ZeRO3 with Ulysses sequence parallelism.

> **Memory note:** `CELOSS_PARALLEL_SIZE=2048` is set automatically by `run_sft.py`
> (via the `env:` section in `base.yaml`). This chunks the CE backward pass to reduce
> peak gradient memory (~18 GB → ~1.2 GB per chunk during backward). It does **not**
> prevent the `lm_head` matmul from materialising the full logit tensor. Do not remove it.
>
> **Library patches required:** the training venv requires three one-time patches to
> `liger_kernel` and `ms-swift` to fix an OOM during in-training eval at 128k context
> (logit matrix does not fit alongside ZeRO-3 gathered params). Run once per machine
> after setting up the venv:
>
> ```bash
> cd /home/ubuntu/Repos/longcontext-baseline && python3 - << 'EOF'
> import pathlib
>
> # Patch 1: liger_kernel — gracefully handle skip_logits=True with no labels (SP eval path)
> p1 = pathlib.Path('.venv/lib/python3.12/site-packages/liger_kernel/transformers/model/qwen3.py')
> t = p1.read_text()
> old = 'raise ValueError("skip_logits is True, but labels and shift_labels are None")'
> new = 'skip_logits = False  # SP eval: labels absent, fall back to logit path'
> assert old in t, 'P1 pattern not found'
> p1.write_text(t.replace(old, new))
> print('patch 1 applied:', p1)
>
> # Patch 2: ms-swift — don't pop labels before model call during eval (SP path)
> p2 = pathlib.Path('.venv/lib/python3.12/site-packages/swift/trainers/seq2seq_trainer.py')
> t = p2.read_text()
> old = "or self.template.sequence_parallel_size > 1) and 'labels' in inputs:"
> new = "or (self.template.sequence_parallel_size > 1 and self.model.training)) and 'labels' in inputs:"
> assert old in t, 'P2 pattern not found'
> p2.write_text(t.replace(old, new))
> print('patch 2 applied:', p2)
>
> # Patch 3: liger_kernel — allow fused CE (skip_logits=True) during eval when labels present
> p3 = pathlib.Path('.venv/lib/python3.12/site-packages/liger_kernel/transformers/model/qwen3.py')
> t = p3.read_text()
> old = 'skip_logits = self.training and (labels is not None or shift_labels is not None)'
> new = 'skip_logits = labels is not None or shift_labels is not None'
> assert old in t, 'P3 pattern not found'
> p3.write_text(t.replace(old, new))
> print('patch 3 applied:', p3)
> EOF
> ```

**Single run:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    ms_swift_train/run_sft.py \
    --config ms_swift_train/configs/64k/clinc150.yaml \
    --paths configs/paths.yaml
```

**Full sweep across 3 EC2 instances (24 jobs total):**

Jobs are pre-assigned to 6 groups of 4 GPUs, balanced by dataset size.
On each instance, run both groups in parallel (background + foreground):

```bash
# On EC2-1
bash ms_swift_train/scripts/run_group.sh ec2_1_gpu0 &
bash ms_swift_train/scripts/run_group.sh ec2_1_gpu4

# On EC2-2
bash ms_swift_train/scripts/run_group.sh ec2_2_gpu0 &
bash ms_swift_train/scripts/run_group.sh ec2_2_gpu4

# On EC2-3
bash ms_swift_train/scripts/run_group.sh ec2_3_gpu0 &
bash ms_swift_train/scripts/run_group.sh ec2_3_gpu4
```

Each group runs its jobs sequentially. A failed job is logged and skipped —
the group continues with the remaining jobs.

Group assignments (sorted large→small so the heaviest job runs first):

| Group | Jobs |
|---|---|
| `ec2_1_gpu0` | `clinc150_128k`, `pop_qa_64k`, `infinite_bench_mc_64k` |
| `ec2_1_gpu4` | `nlu_128k`, `trivia_qa_64k`, `json_kv_64k`, `ms_macro_64k` |
| `ec2_2_gpu0` | `nq_128k`, `nq_64k`, `hotpot_qa_64k`, `ms_macro_128k` |
| `ec2_2_gpu4` | `trivia_qa_128k`, `pop_qa_128k`, `infinite_bench_qa_128k`, `json_kv_128k`, `ruler_mk_uuid_128k` |
| `ec2_3_gpu0` | `trec_coarse_128k`, `nlu_64k`, `trec_coarse_64k` |
| `ec2_3_gpu4` | `hotpot_qa_128k`, `clinc150_64k`, `infinite_bench_mc_128k`, `infinite_bench_qa_64k`, `ruler_mk_uuid_64k` |

> **Running multiple jobs concurrently:** torchrun defaults to port 29500. If you launch
> a second job while another is running, add `--master_port=29501` (or any free port) to
> avoid a "address already in use" error:
> ```bash
> CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 \
>     ms_swift_train/run_sft.py \
>     --config ms_swift_train/configs/64k/hotpot_popqa.yaml \
>     --paths configs/paths.yaml
> ```


### Config structure

`base.yaml` holds all shared hyperparameters (LoRA settings, learning rate,
hardware, logging, etc.). Each per-run config in `64k/` or `128k/` contains
only the delta — `max_length`, `data.datasets`, `output_dir`, and `run_name`.
Fields in the per-run config override the base.

## Inference

> **Separate venv required.** vLLM must run in its own virtual environment, **not** the
> training `.venv`. All vLLM versions ≥ 0.11 link against CUDA 13, breaking a CUDA 12
> training environment. vLLM 0.8.3 (the latest cu12-compatible build) also requires
> `transformers < 5.0`, which conflicts with ms-swift's `transformers ≥ 4.33, < 5.13`.

### Set up the inference venv (one-time)

```bash
# Create a separate venv — do NOT reuse the training .venv
uv venv vllm_inference/.venv --python 3.12
source vllm_inference/.venv/bin/activate

# PyTorch cu128 first, then vLLM, then pin dependencies
uv pip install "torch==2.7.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm==0.8.3"
uv pip install "transformers==4.51.*"   # last 4.x series with Qwen3 support
uv pip install "cachetools<5"           # vLLM 0.8.3 uses LRUCache internals removed in cachetools 5.x

# Install the package (core deps only — no extras)
uv pip install -e "."
```

Switch between venvs:

```bash
source .venv/bin/activate                  # training
source vllm_inference/.venv/bin/activate   # inference
```

### Run inference — full sweep (24 datasets)

```bash
bash vllm_inference/scripts/run_inference_all.sh
```

Paths are resolved from `configs/paths.yaml` automatically. The script:
- skips any dataset whose training checkpoint (`v0-*/best`) is not yet present
- skips any dataset that already has results (`results/finetuned/<dataset>/evaluation.jsonl`)
- logs failed jobs and continues; exits 1 if any failed

To run a subset, comment out lines in the `JOBS` list inside the script.

### Run inference — single dataset

```bash
# Base model
python vllm_inference/inference.py \
    --model-path /mnt/efs/<user>/checkpoints/Qwen/Qwen3-4B-Instruct-2507 \
    --dataset nq_64k \
    --data-dir /mnt/efs/<user>/helmet/longtrain_swift \
    --output-dir /mnt/efs/<user>/results

# Fine-tuned (LoRA adapter — detected automatically via adapter_config.json)
python vllm_inference/inference.py \
    --model-path /mnt/efs/<user>/output/nq_64k/v0-*/best \
    --dataset nq_64k \
    --data-dir /mnt/efs/<user>/helmet/longtrain_swift \
    --output-dir /mnt/efs/<user>/results
```

## Evaluation

```bash
# Score all results
python vllm_inference/evaluate.py /mnt/efs/<user>/results

# Score a single file
python vllm_inference/evaluate.py --file /mnt/efs/<user>/results/finetuned/nq_64k/evaluation.jsonl
```

Scores results using **SubEM** (substring exact match).

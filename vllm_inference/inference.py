import contextlib
import json
import os
import asyncio
import inspect
import argparse
import time
from pathlib import Path
import logging

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest

from utils import process_requests

# Placeholders: replace with your own locations (or pass --data-dir/--output-dir).
DEFAULT_DATA_DIR = "/path/to/base_dir/helmet/longtrain_swift"
DEFAULT_OUTPUT_DIR = "/path/to/results"

logger = logging.getLogger("eval")


def setup_logging(log_file: Path | None = None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

os.environ["VLLM_USE_V1"] = "0"  # V1 engine has a broken LoRA LRU cache in 0.8.3
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "180"


def detect_lora_adapter(model_path: str):
    """
    Check whether model_path is a LoRA adapter directory.
    Returns (base_model, adapter_path, lora_rank) if it is,
    or (model_path, None, None) if it is a plain base model.
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return model_path, None, None

    try:
        with open(adapter_config_path) as f:
            config = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read adapter_config.json: {e}")
        return model_path, None, None

    if "base_model_name_or_path" not in config:
        logger.warning("adapter_config.json has no base_model_name_or_path — treating as base model")
        return model_path, None, None

    base_model = config["base_model_name_or_path"]
    lora_rank = config["r"]
    logger.info(f"LoRA adapter detected at {model_path} (base={base_model}, rank={lora_rank})")
    return base_model, model_path, lora_rank


def initialize_engine(base_model: str, max_input_len: int, lora_rank: int | None) -> AsyncLLMEngine:
    if max_input_len <= 64 * 1024:
        tensor_parallel_size = 1
        max_model_len = 72 * 1024  # headroom above 64k, fits in single-GPU KV cache
    else:
        tensor_parallel_size = 2
        max_model_len = 136 * 1024  # headroom above 128k, split across 2 GPUs

    engine_kwargs = {
        "model": base_model,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 64 * 1024,
        "kv_cache_dtype": "auto",
        "enable_prefix_caching": True,
        "enable_lora": lora_rank is not None,
        "tensor_parallel_size": tensor_parallel_size,
    }

    if lora_rank is not None:
        engine_kwargs["max_lora_rank"] = lora_rank

    engine_args = AsyncEngineArgs(**engine_kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)


async def main(
    base_model: str,
    adapter_path: str | None,
    lora_rank: int | None,
    max_input_length: int,
    input_path: str,
    output_path: str,
    concurrency: int = 4,
):
    engine = initialize_engine(base_model, max_input_length, lora_rank)

    lora_request = None
    if adapter_path:
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=adapter_path,
        )
        logger.info(f"Using LoRA adapter: {adapter_path} (rank={lora_rank})")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(out.suffix + ".tmp")

    with contextlib.suppress(FileNotFoundError):
        tmp_out.unlink()

    try:
        instances = []
        with open(input_path, "r", encoding="utf-8") as f:
            for row in f:
                instances.append(json.loads(row))

        logger.info(f"Loaded {len(instances)} instances from {input_path}")

        await process_requests(
            engine=engine,
            instances=instances,
            output_path=str(tmp_out),
            concurrency=concurrency,
            model_id=base_model,
            max_input_length=max_input_length,
            lora_request=lora_request,
        )

        tmp_out.replace(out)

        logger.info("JOB SUCCEEDED [%s]: %s", Path(adapter_path or base_model).name, out)

    except Exception:
        logger.exception(
            "JOB FAILED [%s]. Partial results kept at: %s",
            Path(adapter_path or base_model).name,
            tmp_out,
        )
        raise

    finally:
        with contextlib.suppress(Exception):
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                ret = shutdown()
                if inspect.isawaitable(ret):
                    await ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on a dataset. "
                    "Accepts a base model path/HF ID or a LoRA adapter directory — "
                    "LoRA is detected automatically from adapter_config.json."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model HF ID / local path, or LoRA adapter directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g. nq_64k, hotpot_qa_128k)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="evaluation",
        choices=["development", "evaluation"],
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Root of the converted swift datasets (see convert_to_swift_messages.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory. "
             "Base model results go to <output-dir>/<model_name>/<dataset>. "
             "LoRA results go to <output-dir>/finetuned/<dataset>.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. Defaults to <output-dir>/inference.log next to the predictions.",
    )

    args = parser.parse_args()

    # Find input file
    input_file = Path(args.data_dir) / args.dataset / f"{args.partition}.jsonl"
    if not input_file.exists():
        dataset_root = Path(args.data_dir)
        raise FileNotFoundError(
            f"Dataset file not found: {input_file}\n"
            f"Available datasets: {[d.name for d in dataset_root.iterdir() if d.is_dir()]}"
        )

    # Infer max context length from dataset name
    if "_64k" in args.dataset:
        max_length = 64 * 1024
    elif "_128k" in args.dataset:
        max_length = 128 * 1024
    else:
        raise ValueError(
            f"Cannot infer max length from dataset name '{args.dataset}'. "
            "Expected a _64k or _128k suffix."
        )

    base_model, adapter_path, lora_rank = detect_lora_adapter(args.model_path)
    is_lora = adapter_path is not None
    if is_lora:
        output_file = Path(args.output_dir) / "finetuned" / args.dataset / f"{args.partition}.jsonl"
    else:
        output_file = Path(args.output_dir) / "baseline" / args.dataset / f"{args.partition}.jsonl"

    log_file = Path(args.log_file) if args.log_file else output_file.with_name("inference.log")
    setup_logging(log_file)

    logger.info("=" * 80)
    logger.info(f"Model:    {args.model_path}  ({'LoRA adapter' if is_lora else 'base model'})")
    logger.info(f"Dataset:  {args.dataset} ({args.partition})")
    logger.info(f"Max ctx:  {max_length // 1024}k")
    logger.info(f"Input:    {input_file}")
    logger.info(f"Output:   {output_file}")
    logger.info("=" * 80)

    start_time = time.time()

    asyncio.run(
        main(
            base_model=base_model,
            adapter_path=adapter_path,
            lora_rank=lora_rank,
            max_input_length=max_length,
            input_path=str(input_file),
            output_path=str(output_file),
            concurrency=args.concurrency,
        )
    )

    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed / 60:.2f} minutes")

"""Launch LoRA SFT via ms-swift.

Usage:
    torchrun --nproc_per_node=4 ms_swift_train/run_sft.py --config ms_swift_train/configs/64k/clinc150_ruler.yaml

Config files support a `base:` key pointing to a parent YAML (relative to the
config file's directory). Fields in the child override fields in the base.

${var} references in string values are substituted from configs/paths.yaml.
Pass --paths to override the default location.
"""

from __future__ import annotations

import os

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths_config import load_paths

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path | None = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    datasets: list[str] = Field(default_factory=list)


class LoRAConfig(BaseModel):
    rank: int = 16
    alpha: int = 16
    target_modules: str = "all-linear"
    dropout: float = 0.0
    bias: str = "none"


class TrainingConfig(BaseModel):
    learning_rate: float = 5e-4
    max_epochs: int = 5
    max_length: int = 65536
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.15
    truncation_strategy: str = "left"
    split_dataset_ratio: float = 0.1
    optim: str = "adamw_torch"


class HardwareConfig(BaseModel):
    torch_dtype: str = "bfloat16"
    attn_impl: str = "flash_attn"
    deepspeed: str = "zero3_offload"
    gradient_checkpointing: bool = True
    sequence_parallel_size: int = 4
    use_liger_kernel: bool = True
    padding_free: bool = True
    max_model_len: int = 131072


class CheckpointingConfig(BaseModel):
    eval_on_start: bool = True
    eval_steps: int = 10
    save_steps: int = 10
    create_checkpoint_symlink: bool = True


class LoggingConfig(BaseModel):
    logging_steps: int = 1
    report_to: str = "tensorboard"


class SFTConfig(BaseModel):
    model_id: str
    output_dir: str
    run_name: str
    data: DataConfig = Field(default_factory=DataConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    checkpointing: CheckpointingConfig = Field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    extra_args: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _expand_vars(obj: Any, vars: dict[str, str]) -> Any:
    """Recursively substitute ${key} references using values from `vars`."""
    if isinstance(obj, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: vars[m.group(1)], obj)
    if isinstance(obj, dict):
        return {k: _expand_vars(v, vars) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(v, vars) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base; override wins on conflict. Dicts are merged recursively."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str, paths_config: str) -> SFTConfig:
    path = Path(config_path).resolve()
    with open(path) as f:
        raw: dict = yaml.safe_load(f)

    base_key = raw.pop("base", None)
    if base_key:
        base_path = (path.parent / base_key).resolve()
        with open(base_path) as f:
            base_raw: dict = yaml.safe_load(f)
        raw = _deep_merge(base_raw, raw)

    raw.pop("paths", None)
    paths_vars = load_paths(paths_config).as_dict()
    raw = _expand_vars(raw, paths_vars)
    return SFTConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def build_swift_args(config: SFTConfig) -> dict[str, Any]:
    t = config.training
    hw = config.hardware
    ck = config.checkpointing
    lg = config.logging
    lora = config.lora

    args: dict[str, Any] = {
        "model": config.model_id,
        "output_dir": config.output_dir,
        "run_name": config.run_name,
        "dataset": config.data.datasets,
        # LoRA
        "tuner_type": "lora",
        "lora_rank": lora.rank,
        "lora_alpha": lora.alpha,
        "target_modules": lora.target_modules,
        "lora_dropout": lora.dropout,
        "lora_bias": lora.bias,
        # training
        "learning_rate": t.learning_rate,
        "max_epochs": t.max_epochs,
        "max_length": t.max_length,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "per_device_eval_batch_size": t.per_device_eval_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "warmup_ratio": t.warmup_ratio,
        "truncation_strategy": t.truncation_strategy,
        "split_dataset_ratio": t.split_dataset_ratio,
        "optim": t.optim,
        # hardware
        "torch_dtype": hw.torch_dtype,
        "attn_impl": hw.attn_impl,
        "deepspeed": hw.deepspeed,
        "gradient_checkpointing": hw.gradient_checkpointing,
        "sequence_parallel_size": hw.sequence_parallel_size,
        "use_liger_kernel": hw.use_liger_kernel,
        "padding_free": hw.padding_free,
        "max_model_len": hw.max_model_len,
        # checkpointing
        "eval_on_start": ck.eval_on_start,
        "eval_steps": ck.eval_steps,
        "save_steps": ck.save_steps,
        "create_checkpoint_symlink": ck.create_checkpoint_symlink,
        # logging
        "logging_steps": lg.logging_steps,
        "report_to": lg.report_to,
    }

    args.update(config.extra_args)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA SFT via ms-swift")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    parser.add_argument("--model_id", help="Override model_id")
    parser.add_argument("--output_dir", help="Override output_dir")
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file (default: train.log in output_dir)")
    cli = parser.parse_args()

    config = load_config(cli.config, cli.paths)

    if cli.model_id:
        config = config.model_copy(update={"model_id": cli.model_id})
    if cli.output_dir:
        config = config.model_copy(update={"output_dir": cli.output_dir})

    log_file = Path(cli.log_file) if cli.log_file else Path(config.output_dir) / "train.log"
    setup_logging(log_file)

    for key, val in config.env.items():
        os.environ.setdefault(key, str(val))
        logger.info(f"env[{key}] = {os.environ[key]}")

    logger.info(f"Model:    {config.model_id}")
    logger.info(f"Output:   {config.output_dir}")
    logger.info(f"Datasets: {config.data.datasets}")
    logger.info(f"max_length: {config.training.max_length}")

    from swift import SftArguments, sft_main

    swift_args = build_swift_args(config)
    sft_main(SftArguments(**swift_args))


if __name__ == "__main__":
    main()

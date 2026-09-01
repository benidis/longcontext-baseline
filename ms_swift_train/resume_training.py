"""Resume training from the latest checkpoint in a run directory.

Reads the existing run's logging.jsonl to determine completed epochs and
compares against num_train_epochs in the config. If incomplete, resumes
from the last/ checkpoint, writing all new checkpoints into the same run dir
so that last/ and best/ symlinks are updated in place.

Usage:
    torchrun --nproc_per_node=4 ms_swift_train/resume_training.py \\
        --run-dir /mnt/efs/kbenidis/output/clinc150_64k/v0-20260721-230854 \\
        --config ms_swift_train/configs/64k/clinc150.yaml \\
        --paths configs/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_completed_epochs(run_dir: Path) -> float:
    logging_jsonl = run_dir / "logging.jsonl"
    if not logging_jsonl.exists():
        raise FileNotFoundError(f"logging.jsonl not found in {run_dir}")
    last_epoch = 0.0
    with open(logging_jsonl) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "epoch" in entry:
                    last_epoch = max(last_epoch, float(entry["epoch"]))
            except (json.JSONDecodeError, ValueError):
                continue
    return last_epoch


def _resolve_checkpoint(run_dir: Path) -> Path:
    # Fall back to last_prev/ if a previous resume attempt already renamed last/.
    for candidate in ("last", "last_prev"):
        link = run_dir / candidate
        if link.is_symlink():
            resolved = link.resolve()
            if resolved.exists():
                return resolved
    raise FileNotFoundError(f"No usable 'last' or 'last_prev' symlink found in {run_dir}")


def main() -> None:
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE))         # ms_swift_train/ (for run_sft)
    sys.path.insert(0, str(_HERE.parent))  # repo root (for paths_config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="Resume ms-swift SFT training from latest checkpoint")
    parser.add_argument("--run-dir", required=True, help="Path to the existing v0-* run directory")
    parser.add_argument("--config", required=True, help="Path to the training config YAML")
    parser.add_argument("--paths", default="configs/paths.yaml", help="Path to paths.yaml")
    cli = parser.parse_args()

    run_dir = Path(cli.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    from run_sft import load_config, build_swift_args, setup_logging

    log_file = run_dir / "train_resume.log"
    setup_logging(log_file)

    config = load_config(cli.config, cli.paths)
    target_epochs = config.training.num_train_epochs

    completed = _read_completed_epochs(run_dir)
    logger.info(f"Completed epochs: {completed:.2f} / {target_epochs}")

    if completed >= target_epochs:
        logger.info("Already complete — nothing to do.")
        sys.exit(0)

    checkpoint = _resolve_checkpoint(run_dir)
    logger.info(f"Resuming from checkpoint: {checkpoint}")

    # Write into the same run dir so last/ and best/ symlinks are updated in place.
    # add_version=False prevents ms-swift from appending another v0-TIMESTAMP
    # subdirectory inside the existing run dir.
    config = config.model_copy(update={
        "output_dir": str(run_dir),
        "extra_args": {
            **config.extra_args,
            "resume_from_checkpoint": str(checkpoint),
            # "resume_only_model": True,  # reset LR scheduler + optimizer; model weights loaded from ckpt
            "add_version": False,
            "eval_on_start": False,
        },
    })

    for key, val in config.env.items():
        os.environ.setdefault(key, str(val))

    logger.info(f"Model:      {config.model_id}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Checkpoint: {checkpoint}")

    # ms-swift uses bare os.symlink (no overwrite) so existing best/ and last/
    # from the initial run would cause FileExistsError in the training finally-block.
    # Rename them to *_prev so ms-swift can recreate them after the resumed run.
    # Only local rank 0 touches the filesystem — other ranks skip this block.
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        for link_name in ("best", "last"):
            link = run_dir / link_name
            if link.is_symlink():
                prev = run_dir / f"{link_name}_prev"
                if prev.is_symlink():
                    prev.unlink()
                link.rename(prev)
                logger.info(f"Renamed {link.name}/ -> {prev.name}/ (target: {prev.resolve()})")

    # DeepSpeed 0.18.x ZeRO-3 CPU-offload resume fix.
    #
    # Root cause: after checkpoint load, cpuadam.lr = 0.0 (end of previous cosine
    # schedule). The resumed lr_scheduler updates a DummyOptimizer, not cpuadam,
    # so every Adam step fires with lr=0 and fp32_flat_groups (the LoRA weights)
    # never change — even though loss and momentum keep moving.
    #
    # Fix part 1: _rigid_load_state_dict — restore cpuadam.lr to initial_lr and
    # capture the cpuadam instance reference for part 2.
    #
    # Fix part 2: AcceleratedScheduler.step — after each scheduler step, write the
    # computed lr directly into the captured cpuadam instance.
    _g_cpuadam = [None]  # filled by part 1, read by part 2

    import deepspeed.runtime.zero.stage3 as _ds_z3
    _orig_rigid_load = _ds_z3.DeepSpeedZeroOptimizer_Stage3._rigid_load_state_dict

    def _fixed_rigid_load(self, state_dict, load_optimizer_states=True):
        _orig_rigid_load(self, state_dict, load_optimizer_states)
        if load_optimizer_states:
            opt_sd = state_dict.get("optimizer_state_dict", {})
            for pg, spg in zip(self.optimizer.param_groups,
                               opt_sd.get("param_groups", [])):
                ilr = spg.get("initial_lr")
                if ilr and ilr > 0:
                    pg["lr"] = ilr
            _g_cpuadam[0] = self.optimizer
        logger.info(f"[DS fix part1] cpuadam captured, lr="
                    f"{[pg['lr'] for pg in self.optimizer.param_groups]}")

    _ds_z3.DeepSpeedZeroOptimizer_Stage3._rigid_load_state_dict = _fixed_rigid_load

    from torch.optim.lr_scheduler import LambdaLR as _LambdaLR
    _orig_lambdalr_step = _LambdaLR.step
    _sched_call_count = [0]

    def _lr_syncing_lambdalr_step(self, *args, **kwargs):
        result = _orig_lambdalr_step(self, *args, **kwargs)
        _sched_call_count[0] += 1
        cpuadam = _g_cpuadam[0]
        if cpuadam is not None:
            try:
                scheduled_lr = self.get_last_lr()[0]
                for pg in cpuadam.param_groups:
                    pg["lr"] = scheduled_lr
                if _sched_call_count[0] <= 3:
                    print(f"[DS fix part2] LambdaLR step {_sched_call_count[0]}: "
                          f"synced lr={scheduled_lr:.6e} to cpuadam", flush=True)
            except Exception as e:
                print(f"[DS fix part2] LambdaLR ERROR step {_sched_call_count[0]}: {e}",
                      flush=True)
        elif _sched_call_count[0] <= 3:
            print(f"[DS fix part2] LambdaLR step {_sched_call_count[0]}: "
                  f"cpuadam not yet captured", flush=True)
        return result

    _LambdaLR.step = _lr_syncing_lambdalr_step

    from swift import SftArguments, sft_main
    sft_main(SftArguments(**build_swift_args(config)))


if __name__ == "__main__":
    main()

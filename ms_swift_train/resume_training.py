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
            "add_version": False,
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

    from swift import SftArguments, sft_main
    sft_main(SftArguments(**build_swift_args(config)))


if __name__ == "__main__":
    main()

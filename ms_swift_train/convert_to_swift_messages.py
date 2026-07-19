#!/usr/bin/env python3
"""Convert local HF datasets into ms-swift messages JSONL format.

This script expects dataset folders like:
  <data_root>/<dataset_name>/development
  <data_root>/<dataset_name>/evaluation

Each record is converted using only:
- input  -> user message content
- output -> assistant message content

By default, if output is a list, only the first non-empty answer is used.
The assistant turn is explicitly marked with loss=true so loss is computed only
on assistant responses.

Usage:
    python ms_swift_train/convert_to_swift_messages.py --paths configs/paths.yaml

Convert a single dataset only:
    python ms_swift_train/convert_to_swift_messages.py \
        --paths configs/paths.yaml --dataset nq_64k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.ipc as _pa_ipc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths_config import load_paths


def pick_output(value: Any, mode: str) -> str:
    """Convert output field to a single assistant string."""
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        if not cleaned:
            return ""
        if mode == "first":
            return cleaned[0]
        return " ||| ".join(cleaned)

    if value is None:
        return ""

    return str(value).strip()


def convert_split(split_path: Path, output_jsonl: Path, output_mode: str) -> tuple[int, int]:
    """Convert one split directory (HF save_to_disk format) to JSONL."""
    arrow_files = sorted(split_path.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {split_path}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for arrow_file in arrow_files:
            with _pa_ipc.open_stream(str(arrow_file)) as reader:
                table = reader.read_all()

            inputs = table.column("input").to_pylist()
            outputs = table.column("output").to_pylist()

            for user_text_raw, output_val in zip(inputs, outputs):
                user_text = str(user_text_raw or "").strip()
                assistant_text = pick_output(output_val, output_mode)

                if not user_text or not assistant_text:
                    skipped += 1
                    continue

                item = {
                    "messages": [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text, "loss": True},
                    ]
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    return written, skipped


def iter_dataset_dirs(data_root: Path) -> list[Path]:
    """Return child dirs that look like dataset containers."""
    dirs: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "development").is_dir() and (p / "evaluation").is_dir():
            dirs.append(p)
    return dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert HF local datasets to ms-swift messages JSONL format."
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.yaml",
        help="Path to paths.yaml config (default: configs/paths.yaml).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional single dataset folder name under data-root (e.g., nq_32k).",
    )
    parser.add_argument(
        "--output-mode",
        choices=["first", "join"],
        default="first",
        help="How to map list output to a single assistant response. Default: first",
    )
    parser.add_argument(
        "--eval-output-mode",
        choices=["first", "join"],
        default=None,
        help=(
            "Optional output mode for evaluation split only. "
            "If not set, uses --output-mode."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    paths = load_paths(args.paths)
    data_root = paths.data_root
    out_root = paths.data_dir

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    if args.dataset:
        dataset_dirs = [data_root / args.dataset]
    else:
        dataset_dirs = iter_dataset_dirs(data_root)

    if not dataset_dirs:
        raise RuntimeError("No dataset directories found with development/evaluation splits.")

    for ds_dir in dataset_dirs:
        if not ds_dir.exists():
            raise FileNotFoundError(f"dataset directory not found: {ds_dir}")

        print(f"\nConverting dataset: {ds_dir.name}")
        for split_name in ["development", "evaluation"]:
            split_path = ds_dir / split_name
            if not split_path.exists():
                print(f"- {split_name}: missing, skipped")
                continue

            out_jsonl = out_root / ds_dir.name / f"{split_name}.jsonl"
            if split_name == "evaluation" and args.eval_output_mode is not None:
                output_mode = args.eval_output_mode
            else:
                output_mode = args.output_mode

            written, skipped = convert_split(split_path, out_jsonl, output_mode)
            print(f"- {split_name}: wrote {written}, skipped {skipped} -> {out_jsonl}")


if __name__ == "__main__":
    main()

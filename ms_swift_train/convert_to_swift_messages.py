#!/usr/bin/env python3
"""Convert local HF datasets into ms-swift messages JSONL format.

Reads from dataset folders like:
  <data_root>/<dataset_name>/development
  <data_root>/<dataset_name>/evaluation

When --metadata is given, the development split is divided into train.jsonl
and val.jsonl using the exact split indices from helmet_metadata.json. For
datasets with list-valued outputs (nq, trivia_qa, hotpot_qa, pop_qa), the
per-sample answer index from target_choice is used as the SFT label.

When --metadata is not given (or a dataset has no entry in the metadata), the
full development split is written to development.jsonl as before.

For evaluation.jsonl, all valid answers are joined with ' ||| ' for
list-answer datasets so that downstream scoring can check against any of them.

Usage:
    python ms_swift_train/convert_to_swift_messages.py \
        --paths configs/paths.yaml \
        --metadata /mnt/efs/out/finetune/data/helmet_metadata.json \
        --seq-len 64k \
        --model-name Qwen3-4B-Instruct-2507

Convert a single dataset only:
    python ms_swift_train/convert_to_swift_messages.py \
        --paths configs/paths.yaml \
        --metadata /mnt/efs/out/finetune/data/helmet_metadata.json \
        --dataset nq_64k
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

# Datasets where output is a list and target_choice selects the SFT label.
# For evaluation these datasets store all answers joined by ' ||| '.
LIST_TARGET_DATASETS = {"nq", "trivia_qa", "hotpot_qa", "pop_qa"}


def _dataset_key(dir_name: str, seq_len: str) -> str | None:
    """'nq_64k' + '64k' -> 'nq', or None if suffix doesn't match."""
    suffix = f"_{seq_len}"
    return dir_name[: -len(suffix)] if dir_name.endswith(suffix) else None


def _pick_one(value: Any, choice_idx: int | None) -> str:
    """Return a single answer string from a raw output value."""
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        if not cleaned:
            return ""
        if choice_idx is not None and choice_idx < len(cleaned):
            return cleaned[choice_idx]
        return cleaned[0]
    if value is None:
        return ""
    return str(value).strip()


def _pick_all(value: Any) -> str:
    """Return all valid answers joined by ' ||| '."""
    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return " ||| ".join(cleaned) if cleaned else ""
    if value is None:
        return ""
    return str(value).strip()


def _read_arrow(split_path: Path) -> tuple[list, list]:
    """Read input and output columns from an arrow split directory."""
    arrow_files = sorted(split_path.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in {split_path}")
    inputs, outputs = [], []
    for af in arrow_files:
        with _pa_ipc.open_stream(str(af)) as reader:
            table = reader.read_all()
        inputs.extend(table.column("input").to_pylist())
        outputs.extend(table.column("output").to_pylist())
    return inputs, outputs


def _write_jsonl(
    path: Path,
    inputs: list,
    outputs: list,
    choices: list[int] | None = None,
    all_answers: bool = False,
) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with path.open("w", encoding="utf-8") as f:
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            user_text = str(inp or "").strip()
            if all_answers:
                assistant_text = _pick_all(out)
            else:
                assistant_text = _pick_one(out, choices[i] if choices is not None else None)
            if not user_text or not assistant_text:
                skipped += 1
                continue
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text, "loss": True},
                ]
            }, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def convert_dataset(
    ds_dir: Path,
    out_root: Path,
    seq_len: str,
    model_name: str,
    val_fraction: str,
    metadata: dict | None,
) -> None:
    ds_key = _dataset_key(ds_dir.name, seq_len)
    is_list_dataset = ds_key in LIST_TARGET_DATASETS
    print(f"\nConverting: {ds_dir.name}  (key={ds_key})")

    # ── Development → train.jsonl + val.jsonl ──────────────────────────────
    dev_path = ds_dir / "development"
    if dev_path.exists():
        dev_inputs, dev_outputs = _read_arrow(dev_path)

        split_info = None
        if metadata and ds_key:
            split_info = (
                metadata.get("train_val_split", {})
                .get(ds_key, {})
                .get(seq_len, {})
                .get(model_name, {})
                .get(val_fraction)
            )

        if split_info is not None:
            train_idx = split_info["train"]
            val_idx = split_info["val"]

            # target_choice: per-sample answer index (only for list-answer datasets)
            tc = None
            if is_list_dataset and metadata:
                tc = (
                    metadata.get("target_choice", {})
                    .get(ds_key, {})
                    .get(seq_len, {})
                    .get(model_name, {})
                )

            for split_name, idx in [("train", train_idx), ("val", val_idx)]:
                sel_inputs  = [dev_inputs[i]  for i in idx]
                sel_outputs = [dev_outputs[i] for i in idx]
                choices = tc[split_name][: len(idx)] if tc else None
                out_jsonl = out_root / ds_dir.name / f"{split_name}.jsonl"
                w, s = _write_jsonl(out_jsonl, sel_inputs, sel_outputs, choices=choices)
                print(f"  {split_name}: wrote {w}, skipped {s}  →  {out_jsonl}")
        else:
            if metadata and ds_key:
                print(f"  WARNING: no train_val_split for {ds_key}/{seq_len}/{model_name} — writing full dev")
            out_jsonl = out_root / ds_dir.name / "development.jsonl"
            w, s = _write_jsonl(out_jsonl, dev_inputs, dev_outputs)
            print(f"  development: wrote {w}, skipped {s}  →  {out_jsonl}")
    else:
        print("  development: missing, skipped")

    # ── Evaluation → evaluation.jsonl ──────────────────────────────────────
    eval_path = ds_dir / "evaluation"
    if eval_path.exists():
        eval_inputs, eval_outputs = _read_arrow(eval_path)
        out_jsonl = out_root / ds_dir.name / "evaluation.jsonl"
        w, s = _write_jsonl(out_jsonl, eval_inputs, eval_outputs, all_answers=is_list_dataset)
        print(f"  evaluation: wrote {w}, skipped {s}  →  {out_jsonl}")
    else:
        print("  evaluation: missing, skipped")


def _iter_dataset_dirs(data_root: Path) -> list[Path]:
    return [
        p for p in sorted(data_root.iterdir())
        if p.is_dir() and (p / "development").is_dir() and (p / "evaluation").is_dir()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--dataset", default=None, help="Single dataset folder name, e.g. nq_64k")
    parser.add_argument("--metadata", default=None, help="Path to helmet_metadata.json")
    parser.add_argument("--seq-len", default="64k", choices=["16k", "32k", "64k", "128k"],
                        help="Sequence-length bucket (used to look up metadata keys). Default: 64k")
    parser.add_argument("--model-name", default="Qwen3-4B-Instruct-2507",
                        help="Model name key in the metadata. Default: Qwen3-4B-Instruct-2507")
    parser.add_argument("--val-fraction", default="0.1",
                        help="Val-fraction key in train_val_split. Default: 0.1")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    data_root = paths.data_root
    out_root = paths.data_dir

    if not data_root.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    metadata = None
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {args.metadata}")

    dataset_dirs = [data_root / args.dataset] if args.dataset else _iter_dataset_dirs(data_root)
    if not dataset_dirs:
        raise RuntimeError("No dataset directories found.")

    for ds_dir in dataset_dirs:
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")
        convert_dataset(ds_dir, out_root, args.seq_len, args.model_name, args.val_fraction, metadata)


if __name__ == "__main__":
    main()

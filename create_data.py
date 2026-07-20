"""Create all 12 helmet datasets at a given max_length.

Downloads the helmet source data (if needed) and prepares the datasets
under <base_dir>/helmet via the keys_values API.

Usage:
    python create_data.py --paths configs/paths.yaml --max-length 64k

--paths:      path to paths.yaml (default: configs/paths.yaml)
--max-length: one of 16k, 32k, 64k, 128k (default: 128k)
--model-id:   HuggingFace model ID for tokenizer (needed by infinite_bench datasets)
"""

import argparse
import importlib.util as _ilu
import os
import pathlib as _pl
import sys
import traceback
from pathlib import Path

# Prevent the HuggingFace fast tokenizer from spawning its own threads, which
# creates RLock objects that cannot be pickled across multiprocessing workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from paths_config import load_paths

# Force all Dataset.map/filter operations to run in the main process.
# Even with num_proc=1, this version of datasets spawns a subprocess via
# iflatmap_unordered, which tries to pickle the function + closure — and
# fails when the closure captures RLock objects (e.g. from the tqdm class).
# Patching iflatmap_unordered to run inline bypasses the pool entirely.
import datasets.utils.py_utils as _py_utils

def _iflatmap_inline(pool, func, *, kwargs_iterable):
    for kwargs in kwargs_iterable:
        yield from func(**kwargs)

_py_utils.iflatmap_unordered = _iflatmap_inline

# Load load_helmet_dev_eval directly from the file to avoid importing
# keys_values/__init__.py which pulls in kvcache (needs newer torch).
_mod_path = _pl.Path("/home/ubuntu/Repos/keys_values/keys_values/data/load_helmet_dev_eval.py")
_spec = _ilu.spec_from_file_location("load_helmet_dev_eval", _mod_path)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_helmet_dev_eval = _mod.load_helmet_dev_eval
download_source_data = _mod.download_source_data

# clinc_oos was renamed to clinc/clinc_oos on HuggingFace Hub; patch the module's
# load_dataset reference so the keys_values code finds it without modification.
_orig_load_dataset = _mod.load_dataset
def _patched_load_dataset(path, *args, **kwargs):
    if path == "clinc_oos":
        path = "clinc/clinc_oos"
    return _orig_load_dataset(path, *args, **kwargs)
_mod.load_dataset = _patched_load_dataset

# Datasets that require a tokenizer for truncation.
NEEDS_TOKENIZER = {"infinite_bench_qa", "infinite_bench_mc"}

RAG_KEYS = [
    "nq", "trivia_qa", "hotpot_qa", "pop_qa",
    "ms_macro", "trec_coarse", "nlu", "clinc150",
    "infinite_bench_qa", "infinite_bench_mc", "json_kv", "ruler_mk_uuid",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        default="configs/paths.yaml",
        help="Path to paths.yaml config (default: configs/paths.yaml).",
    )
    parser.add_argument(
        "--max-length",
        choices=["16k", "32k", "64k", "128k"],
        default="128k",
        help="Context length of the datasets to prepare.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HuggingFace model ID used to load the tokenizer for infinite_bench datasets.",
    )
    args = parser.parse_args()

    paths = load_paths(args.paths)
    max_length = args.max_length
    download_dir = paths.base_dir / "helmet"
    dataset_parent_dir = download_dir / "data"

    if not dataset_parent_dir.is_dir():
        download_source_data(str(download_dir))

    # Load tokenizer once if needed.
    tokenizer = None
    needs_tok = [k for k in RAG_KEYS if k in NEEDS_TOKENIZER]
    if any(not (download_dir / "longtrain" / f"{k}_{max_length}").is_dir() for k in needs_tok):
        if args.model_id is None:
            model_dir = paths.model_dir / "Qwen/Qwen3-4B-Instruct-2507"
            model_id = str(model_dir) if model_dir.exists() else "Qwen/Qwen3-4B-Instruct-2507"
        else:
            model_id = args.model_id
        print(f"Loading tokenizer from {model_id} for infinite_bench datasets...")
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(model_id)
        tokenizer = hf_tok.backend_tokenizer

    failed = []

    for key in RAG_KEYS:
        cache_dir = download_dir / "longtrain" / f"{key}_{max_length}"
        if cache_dir.is_dir():
            print(f"SKIP  {key} @ {max_length} (already cached at {cache_dir})")
            continue

        print(f"\nCREATE {key} @ {max_length} ...")
        try:
            tok_arg = tokenizer if key in NEEDS_TOKENIZER else None
            dev, evl = load_helmet_dev_eval(
                key,
                tokenizer=tok_arg,
                max_length=max_length,
                dataset_parent_dir=str(dataset_parent_dir),
            )
            print(f"  => OK: dev={len(dev)}, eval={len(evl)}")
        except Exception:
            print(f"  => FAILED: {key} @ {max_length}")
            traceback.print_exc()
            failed.append(key)

    if failed:
        print(f"\nFailed: {failed}")
        sys.exit(1)
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()

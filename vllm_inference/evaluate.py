import json
from pathlib import Path
from typing import List


def _eval_rag(responses: List[str], targets: List[List[str]]) -> float:
    scores = []
    for resp, tgt in zip(responses, targets):
        score = 0.0
        for t in tgt:
            if t in resp:
                score = 1.0
                break
        scores.append(score)
    return sum(scores) / len(scores)


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_targets(expected: str) -> List[str]:
    return [t.strip() for t in expected.split("|||") if t.strip()]


def evaluate_file(path: Path) -> dict:
    records = load_jsonl(path)
    responses = [r["output"] for r in records]
    targets = [parse_targets(r["expected"]) for r in records]
    score = _eval_rag(responses, targets)
    return {"file": str(path), "n": len(records), "subEM": round(score, 4)}


def evaluate_all(results_root: Path) -> None:
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found under {results_root}")
        return

    print(f"{'File':<70} {'N':>6}  {'SubEM':>6}")
    print("-" * 86)
    for path in jsonl_files:
        result = evaluate_file(path)
        rel = path.relative_to(results_root)
        print(f"{str(rel):<70} {result['n']:>6}  {result['subEM']:>6.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SubEM on inference JSONL results")
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--file", help="Evaluate a single JSONL file")
    args = parser.parse_args()

    if args.file:
        result = evaluate_file(Path(args.file))
        print(f"SubEM: {result['subEM']:.4f}  (n={result['n']})")
    else:
        evaluate_all(Path(args.results_dir))

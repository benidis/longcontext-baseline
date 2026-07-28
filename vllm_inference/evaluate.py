"""Evaluate inference results with per-dataset metrics.

Metrics are inlined from keys_values to avoid depending on its internal structure.
"""
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Set, Dict


# ---------------------------------------------------------------------------
# Normalisation and extraction (from keys_values/evaluation/response_parser.py)
# ---------------------------------------------------------------------------

def _normalize_string_response(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    prefixes = ("answer", "document", "response", "output", "label", "</think>")
    prefix_re = re.compile(
        r"^(?:" + "|".join(re.escape(p) for p in prefixes) + r")\s*(?:[:=\-]\s*|\s+)?",
        re.IGNORECASE,
    )
    while True:
        new_s = prefix_re.sub("", s, count=1).lstrip()
        if new_s == s:
            break
        s = new_s
    fields_alt = "|".join(re.escape(p) for p in ("answer", "label", "document", "response", "output"))
    cue_re = re.compile(
        rf"(?is)^.*?\b(?:the\s+)?(?:{fields_alt})\b\s*(?:is\b\s*)?(?:[:=\-]\s*|\s+)",
        re.VERBOSE,
    )
    m = cue_re.search(s[:240])
    if m:
        s = s[m.end():].lstrip()
    if len(s) >= 2 and (s[0], s[-1]) in {("`", "`"), ('"', '"'), ("'", "'")}:
        s = s[1:-1].strip()
    return s


def _extract_numbers(text: str) -> List[str]:
    if not text:
        return ["0"]
    number_re = re.compile(
        r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\.(?=\s*(?!\d)))?(?![\w.])",
        re.VERBOSE,
    )
    nums = []
    for m in number_re.findall(text):
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return [str(int(n)) for n in nums] if nums else ["0"]


def _extract_choice_letter(text: str, choices: str = "ABCD", default: str = "") -> str:
    if not text:
        return default
    allowed = set(c.upper() for c in choices if c.strip())
    token_re = re.compile(
        r"(?<![A-Za-z0-9])[\(\[\{\'\"]*(?P<ch>[A-Za-z])[\)\]\}\'\"]*(?:[\.\):,\-])?(?![A-Za-z0-9])",
        re.VERBOSE,
    )
    cue_re = re.compile(r"\b(answer|correct|option|choice|select|selected|pick)\b", re.IGNORECASE)
    candidates = []
    for m in token_re.finditer(text):
        ch = m.group("ch").upper()
        if ch not in allowed:
            continue
        start, end = m.span()
        window = text[max(0, start - 40):min(len(text), end + 40)]
        score = 10 if cue_re.search(window) else 0
        candidates.append((score, start, ch))
    if not candidates:
        return default
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def _extract_value_token(text: str, default: str = "") -> str:
    if not text:
        return default
    hyphen_hex_re = re.compile(r"\b[0-9a-fA-F]{4,}(?:-[0-9a-fA-F]{1,})+\b")
    int_re = re.compile(r"\b\d+\b")
    cue_re = re.compile(r"\b(?:is|are)\b\s*:?\s*", re.IGNORECASE)

    def clean_leading(s):
        s = s.lstrip(" \t\r\n:=-")
        return re.sub(r"^(?:\*\*|__)+", "", s).lstrip()

    def pick_from_chunk(chunk):
        m_bt = re.search(r"`([^`]+)`", chunk)
        if m_bt:
            inside = m_bt.group(1).strip()
            m = hyphen_hex_re.search(inside) or int_re.search(inside)
            if m:
                return m.group(0)
        m = hyphen_hex_re.search(chunk) or int_re.search(chunk)
        return m.group(0) if m else None

    for m in cue_re.finditer(text):
        val = pick_from_chunk(clean_leading(text[m.end():m.end() + 240]))
        if val:
            return val
    all_uuid = hyphen_hex_re.findall(text)
    if all_uuid:
        return all_uuid[-1]
    all_int = int_re.findall(text)
    return all_int[-1] if all_int else default


# ---------------------------------------------------------------------------
# Metrics (from keys_values/evaluation/metrics.py)
# ---------------------------------------------------------------------------

def _normalize_for_tokens(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sub_exact_match(response: str, target_value: str) -> bool:
    response = _normalize_string_response(response).lower()
    target = str(target_value).lower()
    return target in response


def _exact_match(response: str, target_value: str) -> bool:
    return _normalize_string_response(response) == str(target_value)


def _ngram_counts(tokens: list, n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n_f1(response: str, target: str, n: int = 1) -> float:
    response = _normalize_string_response(response) or ""
    resp_tokens = _normalize_for_tokens(response).split()
    targ_tokens = _normalize_for_tokens(target).split()
    resp_ng = _ngram_counts(resp_tokens, n)
    targ_ng = _ngram_counts(targ_tokens, n)
    resp_total = sum(resp_ng.values())
    targ_total = sum(targ_ng.values())
    if resp_total == 0 and targ_total == 0:
        return 1.0
    if resp_total == 0 or targ_total == 0:
        return 0.0
    overlap = sum((resp_ng & targ_ng).values())
    precision = overlap / resp_total
    recall = overlap / targ_total
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def _canon_num(x: str) -> Optional[str]:
    t = str(x).strip()
    if not t:
        return None
    try:
        d = Decimal(t)
    except (InvalidOperation, ValueError):
        return None
    s = format(d.normalize(), "f").rstrip("0").rstrip(".")
    return s if s else "0"


def _ndcg_at_10(pred: List[str], target: List[str], k: int = 10) -> float:
    def dcg(rels):
        return sum((2.0 ** r - 1.0) / math.log2(i + 2) for i, r in enumerate(rels))

    pred_c = [_canon_num(x) for x in pred]
    pred_c = [x for x in pred_c if x is not None][:k]
    target_c = [_canon_num(x) for x in target if _canon_num(x) is not None]

    target_unique: List[str] = []
    seen: Set[str] = set()
    for x in target_c:
        if x not in seen:
            seen.add(x)
            target_unique.append(x)

    if not target_unique or k <= 0:
        return 0.0

    rel_map: Dict[str, float] = {v: float(k - i) for i, v in enumerate(target_unique[:k])}
    used: Set[str] = set()
    pred_rels = []
    for x in pred_c:
        if x in rel_map and x not in used:
            pred_rels.append(rel_map[x])
            used.add(x)
        else:
            pred_rels.append(0.0)

    idcg = dcg([float(k - i) for i in range(min(k, len(target_unique)))])
    return 0.0 if idcg == 0.0 else dcg(pred_rels) / idcg


# ---------------------------------------------------------------------------
# Dataset-level evaluators (from keys_values/evaluation/evaluation.py)
# ---------------------------------------------------------------------------

def _eval_rag(responses: List[str], targets: List[List[str]]) -> float:
    scores = []
    for resp, tgt in zip(responses, targets):
        scores.append(1.0 if any(_sub_exact_match(resp, t) for t in tgt) else 0.0)
    return sum(scores) / len(scores)


def _eval_rerank(responses: List[str], targets: List[str]) -> float:
    scores = []
    for resp, tgt in zip(responses, targets):
        scores.append(_ndcg_at_10(_extract_numbers(resp), [t.strip() for t in tgt.split(">")]))
    return sum(scores) / len(scores)


def _eval_icl(responses: List[str], targets: List[str]) -> float:
    scores = []
    for resp, tgt in zip(responses, targets):
        nums = _extract_numbers(resp)
        best = Counter(nums).most_common(1)[0][0]
        scores.append(1.0 if _exact_match(best, tgt) else 0.0)
    return sum(scores) / len(scores)


def _eval_infinite_qa(responses: List[str], targets: List[str]) -> float:
    scores = [_rouge_n_f1(r, t) for r, t in zip(responses, targets)]
    return sum(scores) / len(scores)


def _eval_infinite_mc(responses: List[str], targets: List[str]) -> float:
    scores = [1.0 if _exact_match(_extract_choice_letter(r), t) else 0.0
              for r, t in zip(responses, targets)]
    return sum(scores) / len(scores)


def _eval_synthetic(responses: List[str], targets: List[str]) -> float:
    scores = []
    for resp, tgt in zip(responses, targets):
        tgt_list = [tgt] if isinstance(tgt, str) else tgt
        resp_val = _extract_value_token(resp)
        resp_list = [resp_val] if isinstance(resp_val, str) else resp_val
        scores.append(min(len(set(resp_list).intersection(tgt_list)) / len(tgt_list), 1))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

DATASET_METRICS = {
    "nq":                ("sub_exact_match", "SubEM"),
    "trivia_qa":         ("sub_exact_match", "SubEM"),
    "pop_qa":            ("sub_exact_match", "SubEM"),
    "hotpot_qa":         ("sub_exact_match", "SubEM"),
    "ms_macro":          ("ndcg_at_10",      "NDCG@10"),
    "trec_coarse":       ("exact_match_icl", "ExactMatch"),
    "nlu":               ("exact_match_icl", "ExactMatch"),
    "clinc150":          ("exact_match_icl", "ExactMatch"),
    "infinite_bench_qa": ("rouge_n_f1",      "ROUGE-1 F1"),
    "infinite_bench_mc": ("infinite_mc",     "ExactMatch"),
    "json_kv":           ("synthetic_value", "SubEM(value)"),
    "ruler_mk_uuid":     ("synthetic_value", "SubEM(value)"),
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _dataset_key(path: Path) -> str:
    name = path.parent.name
    for suffix in ("_128k", "_64k", "_32k", "_16k"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_targets(expected: str) -> List[str]:
    return [t.strip() for t in expected.split("|||") if t.strip()]


def evaluate_file(path: Path) -> dict:
    records = load_jsonl(path)
    responses = [r["output"] for r in records]
    raw_targets = [r["expected"] for r in records]

    dataset_key = _dataset_key(path)
    metric, label = DATASET_METRICS.get(dataset_key, ("sub_exact_match", "SubEM"))

    if metric == "sub_exact_match":
        score = _eval_rag(responses, [parse_targets(t) for t in raw_targets])
    elif metric == "ndcg_at_10":
        score = _eval_rerank(responses, raw_targets)
    elif metric == "exact_match_icl":
        score = _eval_icl(responses, raw_targets)
    elif metric == "rouge_n_f1":
        score = _eval_infinite_qa(responses, raw_targets)
    elif metric == "infinite_mc":
        score = _eval_infinite_mc(responses, raw_targets)
    elif metric == "synthetic_value":
        score = _eval_synthetic(responses, raw_targets)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {"file": str(path), "n": len(records), "score": round(score, 4), "metric": label}


def evaluate_all(results_root: Path) -> None:
    jsonl_files = sorted(results_root.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found under {results_root}")
        return

    print(f"{'File':<70} {'N':>6}  {'Metric':<14}  {'Score':>6}")
    print("-" * 102)
    for path in jsonl_files:
        result = evaluate_file(path)
        rel = path.relative_to(results_root)
        print(f"{str(rel):<70} {result['n']:>6}  {result['metric']:<14}  {result['score']:>6.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate inference results with per-dataset metrics")
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--file", help="Evaluate a single JSONL file")
    args = parser.parse_args()

    if args.file:
        result = evaluate_file(Path(args.file))
        print(f"{result['metric']}: {result['score']:.4f}  (n={result['n']})")
    else:
        evaluate_all(Path(args.results_dir))

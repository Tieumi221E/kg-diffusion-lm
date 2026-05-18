#!/usr/bin/env python3
"""
Unified data processing pipeline for synthetic KG datasets.

Given raw txt files (datasets/<name>/txt) and generation_config.json, this script
produces three folders under datasets/<name>/:

  - pretrain/: full paragraph JSONL + fixed masked eval set (same paragraphs)
  - sft/:     prompt/answer pairs split into train/val
  - qa/:      QA/eval JSONL files (people/job/two-hop/etc.)

No other intermediate folders are kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_utils import build_tokenizer, load_yaml  # noqa: E402

SPECIAL_PROMPT_PREFIX = ""
SPECIAL_PROMPT_SUFFIX = ""
ANSWER_SUFFIX = ""
RNG = random.Random(42)
SENTENCE_SPLIT = re.compile(r"(?<=\.)\s+")

RELATION_KEYWORDS = {
    "father",
    "mother",
    "husband",
    "wife",
    "son",
    "daughter",
    "friend",
    "brother",
    "sister",
    "uncle",
    "aunt",
    "cousin",
    "niece",
    "nephew",
    "spouse",
    "partner",
}

JOB_PATTERNS = [
    re.compile(r"(?:occupation|job|role) is (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
    re.compile(r"works as (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
    re.compile(r"is employed as (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
    re.compile(r"employed as (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
    re.compile(r"serves as (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
    re.compile(r"acts in the role of (?:an? )?([A-Za-z][A-Za-z'-]*)", re.IGNORECASE),
]

NAME_REGEX = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){1,3})\b")
RELATION_REGEXES = [re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in RELATION_KEYWORDS]

DEFAULT_VAL_SIZE = 100
DEFAULT_FIXED_EVAL = 200

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process synthetic KG dataset into pretrain/SFT/QA JSONL.")
    parser.add_argument("--dataset-name", required=True, help="e.g., 50000+500x10")
    parser.add_argument(
        "--source-subdir",
        default="txt",
        help="relative subdirectory containing raw txt (under datasets/<name>/). Use '.' if files live at root.",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="configs/train_mdm.yaml",
        help="YAML config used to build tokenizer (defaults to pretrain config).",
    )
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--fixed-eval-samples", type=int, default=DEFAULT_FIXED_EVAL)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def read_blocks(path: Path) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                current.append(line.rstrip("\n"))
            else:
                if current:
                    blocks.append("\n".join(current))
                    current = []
        if current:
            blocks.append("\n".join(current))
    return blocks

def split_sentences(text: str) -> List[str]:
    pieces = [seg.strip() for seg in SENTENCE_SPLIT.split(text) if seg.strip()]
    if not pieces:
        pieces = [seg.strip() for seg in text.split(".") if seg.strip()]
    return pieces or [text.strip()]

def _ensure_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return ""
    return sentence if sentence.endswith(".") else f"{sentence}."

def _token_overlaps_span(token_span: Tuple[int, int], candidate_spans: Sequence[Tuple[int, int]]) -> bool:
    start, end = token_span
    if end <= start:
        return False
    for span_start, span_end in candidate_spans:
        if start < span_end and end > span_start:
            return True
    return False

def _token_has_letters(segment: str) -> bool:
    return any(ch.isalnum() for ch in segment)

def _find_first_name_span(text: str) -> Tuple[int, int] | None:
    match = NAME_REGEX.search(text)
    if not match:
        return None
    return match.span(1 if match.lastindex else 0)

def _find_last_job_span(text: str) -> Tuple[int, int] | None:
    spans: List[Tuple[int, int]] = []
    for pattern in JOB_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex:
                spans.append(match.span(1))
    if not spans:
        return None
    spans.sort(key=lambda span: span[0])
    return spans[-1]

def _prepare_five_sentence_text(entry_text: str) -> str | None:
    sentences = split_sentences(entry_text)
    if len(sentences) < 5:
        return None
    trimmed = sentences[:5]
    normalized = [_ensure_sentence(sent) for sent in trimmed if sent.strip()]
    if len(normalized) < 5:
        return None
    sample_text = " ".join(normalized).strip()
    return sample_text or None

def split_question_answer(block: str) -> Tuple[str, str] | None:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return None
    if len(lines) == 1:
        text = lines[0]
        q_idx = text.find("?")
        if q_idx == -1:
            return None
        question = text[: q_idx + 1].strip()
        answer = text[q_idx + 1 :].strip()
        if not answer:
            return None
        return question, answer
    question = lines[0]
    answer = lines[1] if len(lines) > 1 else ""
    if not answer:
        return None
    return question, answer

def format_prompt(question: str) -> str:
    parts = []
    if SPECIAL_PROMPT_PREFIX:
        parts.append(SPECIAL_PROMPT_PREFIX)
    parts.append(question.strip())
    if SPECIAL_PROMPT_SUFFIX:
        parts.append(SPECIAL_PROMPT_SUFFIX)
    return "\n".join(part for part in parts if part)

def format_answer(answer: str) -> str:
    return f"{answer}{ANSWER_SUFFIX}"

def count_tokens(text: str) -> int:
    return len([tok for tok in text.replace("\n", " ").split() if tok])

def build_uid(prompt: str, answer: str) -> str:
    data = (prompt + "\n" + answer).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:16]

def classify_question(question: str) -> str:
    lower = question.lower()
    if any(key in lower for key in ["occupation", "work as", "job", "employed as"]):
        return "job"
    if "whose" in lower:
        return "forward"
    if lower.startswith("who is") or lower.startswith("who serves") or lower.startswith("who acts"):
        return "reverse"
    return "unknown"

def derive_prefixes(dataset_name: str) -> Tuple[str, str]:
    primary = dataset_name.split("+", 1)[0]
    base_prefix = f"{primary}x"
    pretrain_prefix = f"{primary}+500x"
    return base_prefix, pretrain_prefix

def write_jsonl(path: Path, entries: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f_out:
        for item in entries:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

def split_train_val(entries: Sequence[Dict], val_size: int, rng: random.Random) -> Tuple[List[Dict], List[Dict]]:
    if not entries:
        return [], []
    shuffled = list(entries)
    rng.shuffle(shuffled)
    val_size = min(val_size, len(shuffled))
    val_entries = shuffled[:val_size]
    train_entries = shuffled[val_size:]
    if not train_entries:
        train_entries = val_entries[:-1]
        val_entries = val_entries[-1:]
    return train_entries, val_entries

def convert_standard_qa(src_path: Path, answer_type_hint: str | None = None) -> List[Dict]:
    entries: List[Dict] = []
    for block in read_blocks(src_path):
        qa = split_question_answer(block)
        if not qa:
            continue
        question, answer = qa
        prompt = format_prompt(question)
        formatted_answer = format_answer(answer)
        uid = build_uid(prompt, formatted_answer)
        q_type = classify_question(answer_type_hint or question)
        entry = {
            "uid": uid,
            "prompt": prompt,
            "answer": formatted_answer,
            "prompt_length": count_tokens(prompt),
            "answer_length": count_tokens(formatted_answer),
            "answer_type": "job" if q_type == "job" else "name",
        }
        entries.append(entry)
    return entries

def convert_pretrain(src_path: Path) -> List[Dict]:
    entries: List[Dict] = []
    for block in read_blocks(src_path):
        text = block.strip()
        if not text:
            continue
        sentences = split_sentences(text)
        normalized = [_ensure_sentence(s) for s in sentences if s.strip()]
        if not normalized:
            continue
        entry = {
            "uid": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
            "raw_text": " ".join(normalized).strip(),
            "sentence_count": len(normalized),
        }
        entries.append(entry)
    return entries

def expand_pretrain(entries: List[Dict], rng: random.Random, subset_repeats: int = 2) -> List[Dict]:
    """
    Expands each raw entry into 1 + 1 + subset_repeats samples:
      - Part 1 Raw: one copy of the full paragraph
      - Part 2 Single: one randomly sampled sentence
      - Part 3 Subsets: random subsets of length 2..N-1, repeated subset_repeats times
    """
    expanded: List[Dict] = []
    for entry in entries:
        text = entry["raw_text"]
        sentences = split_sentences(text)
        normalized = [_ensure_sentence(s) for s in sentences if s.strip()]
        n = len(normalized)
        if n == 0:
            continue
        base_uid = entry["uid"]

        # Part 1: Raw
        raw_entry = dict(entry)
        raw_entry["raw_text"] = " ".join(normalized).strip()
        raw_entry["sentence_count"] = n
        expanded.append(raw_entry)

        # Part 2: Single sentence
        idx = rng.randrange(n)
        single_entry = {
            "uid": f"{base_uid}-s{idx}",
            "raw_text": normalized[idx],
            "sentence_count": 1,
        }
        expanded.append(single_entry)

        # Part 3: Subsets (two repeats by default)
        if n >= 3:  # need at least 3 sentences to sample a subset of size 2..n-1
            for rep in range(subset_repeats):
                k = rng.randint(2, n - 1)
                chosen_idx = sorted(rng.sample(range(n), k))
                subset_text = " ".join(normalized[i] for i in chosen_idx).strip()
                sub_entry = {
                    "uid": f"{base_uid}-sub{rep}",
                    "raw_text": subset_text,
                    "sentence_count": k,
                }
                expanded.append(sub_entry)
    return expanded

def convert_in_context(src_path: Path) -> Tuple[List[Dict], List[Dict]]:
    with_entries: List[Dict] = []
    without_entries: List[Dict] = []
    for block in read_blocks(src_path):
        qa = split_question_answer(block)
        if not qa:
            continue
        context, answer = qa
        full_prompt = format_prompt(context)
        last_sentence = context.strip()
        if ". " in last_sentence:
            last_sentence = last_sentence.rsplit(". ", 1)[-1]
        trimmed_prompt = format_prompt(last_sentence)
        formatted_answer = format_answer(answer)
        uid_full = build_uid(full_prompt, formatted_answer)
        uid_trim = build_uid(trimmed_prompt, formatted_answer)
        entry_full = {
            "uid": uid_full,
            "prompt": full_prompt,
            "answer": formatted_answer,
            "prompt_length": count_tokens(full_prompt),
            "answer_length": count_tokens(formatted_answer),
        }
        entry_trim = {
            "uid": uid_trim,
            "prompt": trimmed_prompt,
            "answer": formatted_answer,
            "prompt_length": count_tokens(trimmed_prompt),
            "answer_length": count_tokens(formatted_answer),
        }
        with_entries.append(entry_full)
        without_entries.append(entry_trim)
    return with_entries, without_entries

def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name
    root = Path("datasets") / dataset_name
    if args.source_subdir in (".", ""):
        raw_dir = root
    else:
        raw_dir = root / args.source_subdir
    if not raw_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {raw_dir}")

    pretrain_dir = root / "pretrain"
    sft_dir = root / "sft"
    qa_dir = root / "qa"
    for path in (pretrain_dir, sft_dir, qa_dir):
        path.mkdir(parents=True, exist_ok=True)

    tokenizer_cfg = load_yaml(Path(args.tokenizer_config))["tokenizer"]

    base_prefix, pretrain_prefix = derive_prefixes(dataset_name)

    # --- Pretrain ---
    pretrain_txt = raw_dir / f"synthetic_logic_mixed_{pretrain_prefix}_template_10.txt"
    pretrain_entries = convert_pretrain(pretrain_txt)
    pretrain_json = pretrain_dir / f"synthetic_logic_mixed_{pretrain_prefix}_template_10.jsonl"
    write_jsonl(pretrain_json, pretrain_entries)

    expanded_entries = expand_pretrain(pretrain_entries, RNG, subset_repeats=2)
    expanded_json = pretrain_dir / f"synthetic_logic_mixed_{pretrain_prefix}_template_10_expanded.jsonl"
    write_jsonl(expanded_json, expanded_entries)
    # fixed eval removed
    
    # --- SFT prompt/answer ---
    base_txt = raw_dir / f"ft_qa_pair_data_{base_prefix}_template_10.txt"
    sft_entries = convert_standard_qa(base_txt)
    train_entries, val_entries = split_train_val(sft_entries, args.val_size, RNG)
    write_jsonl(sft_dir / f"ft_qa_pair_data_{base_prefix}_template_10_train.jsonl", train_entries)
    write_jsonl(sft_dir / f"ft_qa_pair_data_{base_prefix}_template_10_val.jsonl", val_entries)

    # --- QA splits ---
    qa_specs = [
        ("qa_completion_people_500x_template_10.txt", "qa_people_reverse.jsonl", "people_reverse"),
        ("qa_completion_people_forward_500x_template_10.txt", "qa_people_forward.jsonl", "people_forward"),
        ("qa_completion_job_500x_template_10.txt", "qa_job.jsonl", "job"),
        ("qa_completion_complex_job_500x_template_10.txt", "qa_two_hop.jsonl", "two_hop"),
        ("qa_inversion_500x_template_10.txt", "qa_inversion.jsonl", "inversion"),
        ("qa_symmetry_500x_template_10.txt", "qa_symmetry.jsonl", "symmetry"),
    ]
    for src_name, dst_name, qtype in qa_specs:
        entries = convert_standard_qa(raw_dir / src_name, answer_type_hint=qtype)
        write_jsonl(qa_dir / dst_name, entries)

    incontext_specs = [
        ("qa_in_context_inversion_300.txt", "incontext_inversion_with_ft.jsonl", "incontext_inversion_without_ft.jsonl"),
        ("qa_in_context_symmetry_300.txt", "incontext_symmetry_with_ft.jsonl", "incontext_symmetry_without_ft.jsonl"),
    ]
    for src_name, dst_with, dst_without in incontext_specs:
        with_entries, without_entries = convert_in_context(raw_dir / src_name)
        write_jsonl(qa_dir / dst_with, with_entries)
        write_jsonl(qa_dir / dst_without, without_entries)

    print(f"Pretrain JSONL written to {pretrain_json}")
    print(f"Pretrain expanded JSONL written to {expanded_json}")
    print(f"SFT train/val written under {sft_dir}")
    print(f"QA splits written under {qa_dir}")

if __name__ == "__main__":
    main()

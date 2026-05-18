"""
Pre-training dataset and collator utilities for the synthetic logic corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from diffusion_core.masking import apply_diffusion_mask as apply_random_mask


@dataclass(frozen=True)
class PretrainSample:
    uid: str
    text: str
    sentence_count: int


class SyntheticLogicPretrainDataset(Dataset):
    """
    Thin JSONL-backed dataset. Each item contains the original paragraph text plus metadata.
    """

    def __init__(
        self,
        data_path: Path,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Pre-training dataset not found: {self.data_path}")
        self._samples: List[PretrainSample] = []
        self._load_samples()

    def _load_samples(self) -> None:
        suffix = self.data_path.suffix.lower()
        if suffix == ".jsonl":
            self._load_from_jsonl()
        else:
            self._load_from_text()

    def _load_from_jsonl(self) -> None:
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                sentences = obj.get("sentences")
                if isinstance(sentences, list) and sentences:
                    count = len(sentences)
                else:
                    count = int(obj.get("sentence_count", 0) or 0)
                    raw = obj.get("raw_text", "")
                    if count <= 0 and raw:
                        pieces = [s.strip() for s in re.split(r"(?<=\.)\s+", raw) if s.strip()]
                        count = len(pieces)
                        sentences = pieces
                    elif not sentences:
                        sentences = []

                raw_text = obj.get("raw_text") or " ".join(sentences)
                uid = str(obj.get("paragraph_id", obj.get("uid")))
                sample = PretrainSample(uid=uid, text=raw_text.replace("\n", " ").strip(), sentence_count=count)
                self._samples.append(sample)

    def _load_from_text(self) -> None:
        raw_text = self.data_path.read_text(encoding="utf-8")
        blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
        sha1 = hashlib.sha1
        split_pattern = re.compile(r"(?<=\.)\s+")
        for idx, block in enumerate(blocks):
            sentences = [s.strip() for s in split_pattern.split(block) if s.strip()]
            count = len(sentences)
            uid_source = block.encode("utf-8")
            uid = sha1(uid_source).hexdigest()[:16]
            self._samples.append(
                PretrainSample(uid=uid, text=block.replace("\n", " ").strip(), sentence_count=count)
            )
        if not self._samples:
            raise ValueError(f"No samples were found in {self.data_path}.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> PretrainSample:
        return self._samples[idx]


class PretrainMaskCollator:
    """
    Collator applying Bernoulli masking with batch-level p_mask and optional random cropping.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_token: str = "[MASK]",
        random_crop_prob: float = 0.01,
        eps: float = 1e-3,
        prepend_bos: bool = False,
        append_eos: bool = False,
        special_token_ids: Optional[Sequence[int]] = None,
        pad_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        mask_content_only: bool = False,
        content_stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        self.random_crop_prob = float(random_crop_prob)
        if not (0.0 <= self.random_crop_prob <= 1.0):
            raise ValueError("random_crop_prob must lie within [0, 1].")
        self.eps = float(eps)
        if not (0.0 <= self.eps < 1.0):
            raise ValueError("eps must lie within [0, 1).")
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.generator = generator

        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_token_id is None or mask_token_id < 0:
            raise ValueError(
                f"Tokenizer does not contain the mask token {mask_token}. "
                "Call tokenizer.add_special_tokens({'additional_special_tokens': ['[MASK]']}) first."
            )
        self.mask_token_id = int(mask_token_id)

        self.bos_token_id = tokenizer.bos_token_id if prepend_bos else None
        self.eos_token_id = tokenizer.eos_token_id if append_eos else None

        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif tokenizer.pad_token_id is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        elif tokenizer.eos_token_id is not None:
            # GPT-2 has no dedicated PAD token; fall back to EOS
            self.pad_token_id = int(tokenizer.eos_token_id)
        else:
            self.pad_token_id = 0

        special_ids = {self.pad_token_id, self.mask_token_id}
        if special_token_ids:
            special_ids.update(int(s) for s in special_token_ids)
        self.special_token_ids = special_ids
        self.mask_content_only = bool(mask_content_only)
        self.stopword_ids: torch.Tensor
        if self.mask_content_only:
            default_stops = content_stopwords or [
                "is",
                "are",
                "as",
                "the",
                "of",
                "to",
                "and",
                "a",
                "an",
                "in",
                "on",
                "at",
                "by",
                "for",
                "with",
                "from",
                "that",
                "this",
                "was",
                "were",
                "be",
                "been",
                "do",
                "did",
                "does",
                "has",
                "have",
                "had",
                "his",
                "her",
                "their",
            ]
            ids: List[int] = []
            for w in default_stops:
                toks = tokenizer.tokenize(w)
                if len(toks) == 1:
                    tid = tokenizer.convert_tokens_to_ids(toks[0])
                    if tid is not None and tid >= 0:
                        ids.append(int(tid))
            self.stopword_ids = torch.tensor(ids, dtype=torch.long)
        else:
            self.stopword_ids = torch.empty(0, dtype=torch.long)

    @staticmethod
    def _forward_process(
        clean_input: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        mask_token_id: int,
        eps: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return apply_random_mask(
            clean_input=clean_input,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            mask_token_id=mask_token_id,
            eps=eps,
            generator=generator,
        )

    def _encode_text(self, text: str) -> Tuple[List[int], int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        original_length = len(tokens)
        if self.prepend_bos and self.bos_token_id is not None:
            tokens = [self.bos_token_id] + tokens
        if self.append_eos and self.eos_token_id is not None:
            tokens = tokens + [self.eos_token_id]
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        return tokens, original_length

    def _maybe_random_crop(self, tokens: List[int]) -> Tuple[List[int], bool]:
        if not tokens:
            return tokens, False
        if random.random() >= self.random_crop_prob:
            return tokens, False
        crop_len = random.randint(1, len(tokens))
        return tokens[:crop_len], True

    def __call__(self, batch: Iterable[PretrainSample]) -> dict:
        simple_batch = list(batch)
        if not simple_batch:
            raise ValueError("batch must not be empty.")

        encoded_items: List[List[int]] = []
        orig_lengths: List[int] = []
        crop_flags: List[bool] = []
        uids: List[str] = []
        for sample in simple_batch:
            tokens, orig_len = self._encode_text(sample.text)
            tokens, cropped = self._maybe_random_crop(tokens)
            encoded_items.append(tokens)
            orig_lengths.append(orig_len)
            crop_flags.append(cropped)
            uids.append(sample.uid)

        batch_size = len(encoded_items)
        device = torch.device("cpu")
        max_len = self.max_length

        clean_input = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for i, tokens in enumerate(encoded_items):
            seq_len = min(len(tokens), max_len)
            if seq_len == 0:
                continue
            clean_input[i, :seq_len] = torch.tensor(tokens[:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = True

        special_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for special_id in self.special_token_ids:
            special_mask |= clean_input.eq(special_id)
        valid_positions = attention_mask & ~special_mask
        if self.mask_content_only and self.stopword_ids.numel() > 0:
            stop_mask = torch.isin(clean_input, self.stopword_ids.to(device))
            valid_positions = valid_positions & ~stop_mask

        noisy_input_ids, target_ids, masked_positions, p_scalar = self._forward_process(
            clean_input,
            attention_mask,
            valid_positions,
            self.mask_token_id,
            self.eps,
            self.generator,
        )
        masked_token_counts = masked_positions.sum(dim=1).to(torch.long)

        result = {
            "input_ids": noisy_input_ids,
            "attention_mask": attention_mask.long(),
            "target_ids": target_ids,
            "masked_positions": masked_positions,
            "masked_token_counts": masked_token_counts,
            "p_mask_scalar": p_scalar,
            "orig_lengths": torch.tensor(orig_lengths, dtype=torch.long, device=device),
            "random_crop_applied": torch.tensor(crop_flags, dtype=torch.bool, device=device),
            "uids": uids,
        }
        return result


class FixedMaskPretrainEvalDataset(Dataset):
    """
    Dataset that serves pre-tokenised, fixed-mask samples for evaluation.
    """

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Fixed eval dataset not found: {self.jsonl_path}")
        self._samples: List[Dict[str, List[int]]] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                required = ("input_ids", "attention_mask", "target_ids")
                if not all(k in obj for k in required):
                    continue
                self._samples.append({
                    "input_ids": list(obj["input_ids"]),
                    "attention_mask": list(obj["attention_mask"]),
                    "target_ids": list(obj["target_ids"]),
                    "masked_positions": list(obj.get("masked_positions", [])),
                })
        if not self._samples:
            raise ValueError(f"No samples loaded from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long),
            "masked_positions": torch.tensor(sample["masked_positions"], dtype=torch.bool),
        }


@dataclass(frozen=True)
class ClozeQASample:
    uid: str
    prefix_full: str
    prefix_question: str
    answer: str
    suffix: str


class ClozeQADataset(Dataset):
    """
    Dataset for cloze QA records produced by scripts/convert_qa_to_cloze.py.
    """

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Cloze dataset not found: {self.jsonl_path}")
        self._samples: List[ClozeQASample] = []
        self._load()

    @staticmethod
    def _split_cloze(text: str) -> Tuple[str, str, str]:
        if "（" not in text or "）" not in text:
            raise ValueError(f"Cloze text missing parentheses: {text}")
        prefix, rest = text.split("（", 1)
        answer, suffix = rest.split("）", 1)
        return prefix, answer, suffix

    @staticmethod
    def _question_only(prefix: str) -> str:
        separators = ["。", ".", "?", "！", "!", "？"]
        idx = -1
        sep_len = 0
        for sep in separators:
            pos = prefix.rfind(sep)
            if pos > idx:
                idx = pos
                sep_len = len(sep)
        if idx == -1:
            return prefix
        return prefix[idx + sep_len :].lstrip()

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                filled = obj.get("cloze_filled")
                if not filled:
                    continue
                prefix, answer, suffix = self._split_cloze(filled)
                sample = ClozeQASample(
                    uid=str(obj.get("uid", "")),
                    prefix_full=prefix,
                    prefix_question=self._question_only(prefix),
                    answer=answer,
                    suffix=suffix,
                )
                self._samples.append(sample)
        if not self._samples:
            raise ValueError(f"No cloze samples were found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClozeQASample:
        return self._samples[idx]


class ClozeFillCollator:
    """
    Collator that replaces the answer span inside parentheses with [MASK] tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_token: str = "[MASK]",
        pad_token_id: Optional[int] = None,
        use_question_prefix: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_token_id is None or mask_token_id < 0:
            raise ValueError(f"Tokenizer missing mask token {mask_token}")
        self.mask_token_id = int(mask_token_id)
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif tokenizer.pad_token_id is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        else:
            self.pad_token_id = tokenizer.eos_token_id or 0
        self.use_question_prefix = use_question_prefix

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _trim(
        self,
        prefix_ids: List[int],
        answer_ids: List[int],
        suffix_ids: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        total = len(prefix_ids) + len(answer_ids) + len(suffix_ids)
        if total <= self.max_length:
            return prefix_ids, answer_ids, suffix_ids
        overflow = total - self.max_length
        if len(suffix_ids) > 0:
            trim = min(len(suffix_ids), overflow)
            suffix_ids = suffix_ids[:-trim] if trim < len(suffix_ids) else []
            overflow -= trim
        if overflow > 0 and len(prefix_ids) > 0:
            trim = min(len(prefix_ids), overflow)
            prefix_ids = prefix_ids[trim:]
            overflow -= trim
        if overflow > 0 and len(answer_ids) > 1:
            trim = min(len(answer_ids) - 1, overflow)
            answer_ids = answer_ids[:-trim]
        return prefix_ids, answer_ids, suffix_ids

    def __call__(self, batch: Iterable[ClozeQASample]) -> Dict[str, torch.Tensor]:
        samples = list(batch)
        if not samples:
            raise ValueError("batch must not be empty.")
        batch_size = len(samples)
        device = torch.device("cpu")
        input_ids = torch.full((batch_size, self.max_length), self.pad_token_id, dtype=torch.long, device=device)
        target_ids = torch.full((batch_size, self.max_length), -100, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool, device=device)
        masked_positions = torch.zeros((batch_size, self.max_length), dtype=torch.bool, device=device)
        p_mask = torch.ones((batch_size, self.max_length), dtype=torch.float32, device=device)

        for idx, sample in enumerate(samples):
            prefix_text = sample.prefix_question if self.use_question_prefix else sample.prefix_full
            prefix_ids = self._encode(f"{prefix_text}（")
            suffix_ids = self._encode(f"）{sample.suffix}")
            answer_ids = self._encode(sample.answer)
            if not answer_ids:
                answer_ids = [self.mask_token_id]
            prefix_ids, answer_ids, suffix_ids = self._trim(prefix_ids, answer_ids, suffix_ids)
            mask_count = len(answer_ids)
            seq_prefix = prefix_ids
            seq_suffix = suffix_ids
            full_input = seq_prefix + [self.mask_token_id] * mask_count + seq_suffix
            full_target = seq_prefix + answer_ids + seq_suffix
            seq_len = min(len(full_input), self.max_length)
            input_ids[idx, :seq_len] = torch.tensor(full_input[:seq_len], dtype=torch.long)
            target_ids[idx, :seq_len] = torch.tensor(full_target[:seq_len], dtype=torch.long)
            attention_mask[idx, :seq_len] = True
            answer_start = len(seq_prefix)
            answer_end = min(answer_start + mask_count, self.max_length)
            masked_positions[idx, answer_start:answer_end] = True
            target_ids[idx, :answer_start] = -100
            if answer_end < seq_len:
                target_ids[idx, answer_end:seq_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.long(),
            "target_ids": target_ids,
            "masked_positions": masked_positions,
            "p_mask": p_mask,
            "uids": [s.uid for s in samples],
        }

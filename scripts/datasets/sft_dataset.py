"""Cloze-style SFT dataset + masking collator."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, GPT2TokenizerFast

from diffusion_core.masking import apply_diffusion_mask as apply_random_mask


@dataclass(frozen=True)
class ClozeSample:
    uid: str
    text: str
    answer: str
    question_type: str
    logic_tags: Sequence[str]
    context: Sequence[str]


class ClozeDataset(Dataset):
    """Loads cloze sentences (full sentence with answer in parentheses)."""

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if jsonl_path is None and records is None:
            raise ValueError("Either jsonl_path or records must be provided.")
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        if self.jsonl_path is not None and not self.jsonl_path.exists():
            raise FileNotFoundError(f"Cloze dataset not found: {self.jsonl_path}")
        self._samples: List[ClozeSample] = []
        if records is not None:
            self._load_from_records(records)
        else:
            assert self.jsonl_path is not None
            self._load_from_path(self.jsonl_path)
        if not self._samples:
            source = self.jsonl_path if self.jsonl_path is not None else "provided records"
            raise ValueError(f"No cloze samples were found in {source}.")

    def _load_from_path(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        self._load_from_records(records)

    def _load_from_records(self, records: List[Dict[str, Any]]) -> None:
        for obj in records:
            text = obj.get("cloze_filled") or obj.get("text")
            if not text:
                continue
            answer = self._extract_answer(text)
            sample = ClozeSample(
                uid=str(obj.get("uid", len(self._samples))),
                text=text,
                answer=answer,
                question_type=obj.get("question_type", "unknown"),
                logic_tags=list(obj.get("logic_tags", [])),
                context=list(obj.get("context", [])),
            )
            self._samples.append(sample)

    @staticmethod
    def _extract_answer(text: str) -> str:
        if "（" in text and "）" in text:
            try:
                return text.split("（", 1)[1].split("）", 1)[0]
            except ValueError:
                return ""
        return ""

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ClozeSample:
        return self._samples[idx]


class ClozeMaskCollator:
    """Masks only the answer span inside parentheses for diffusion-style SFT."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_token: str = "[MASK]",
        masking_mode: str = "random",
        fixed_mask_option: Optional[Any] = None,
        eps: float = 1e-3,
        pad_token_id: Optional[int] = None,
        full_mask_probability: float = 0.0,
        partial_mask_min: Optional[int] = None,
        partial_mask_max: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        self.masking_mode = masking_mode
        self.fixed_mask_option = fixed_mask_option
        self.eps = float(eps)
        if not (0.0 <= self.eps < 1.0):
            raise ValueError("eps must lie within [0, 1).")
        self.full_mask_probability = float(full_mask_probability)
        if not (0.0 <= self.full_mask_probability <= 1.0):
            raise ValueError("full_mask_probability must be in [0, 1].")
        if (partial_mask_min is None) ^ (partial_mask_max is None):
            raise ValueError("partial_mask_min and partial_mask_max must be both provided or both None.")
        self.partial_mask_min = int(partial_mask_min) if partial_mask_min is not None else None
        self.partial_mask_max = int(partial_mask_max) if partial_mask_max is not None else None
        if self.partial_mask_min is not None and self.partial_mask_min <= 0:
            raise ValueError("partial_mask_min must be positive.")
        if self.partial_mask_max is not None and self.partial_mask_max < self.partial_mask_min:
            raise ValueError("partial_mask_max must be >= partial_mask_min.")
        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_id is None or mask_id < 0:
            raise ValueError(f"Tokenizer missing mask token {mask_token}")
        self.mask_token_id = int(mask_id)
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif tokenizer.pad_token_id is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        else:
            self.pad_token_id = tokenizer.eos_token_id or 0
        self.generator = generator

    @staticmethod
    def _split_cloze_sentence(text: str) -> Tuple[str, str, str]:
        if "（" not in text or "）" not in text:
            raise ValueError(f"Cloze sentence missing parentheses: {text}")
        prefix, rest = text.split("（", 1)
        answer, suffix = rest.split("）", 1)
        return prefix, answer, suffix

    def _encode_segments(
        self,
        prefix: str,
        answer: str,
        suffix: str,
    ) -> Tuple[List[int], List[int], List[int]]:
        prefix_ids = self.tokenizer.encode(f"{prefix}（", add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(f"）{suffix}", add_special_tokens=False)
        return prefix_ids, answer_ids, suffix_ids

    def _trim_segments(
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

    def _select_mask(self, answer_len: int) -> Tuple[torch.Tensor, float]:
        device = torch.device("cpu")
        if answer_len <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device), 0.0
        generator = self.generator
        rand_val = random.random() if generator is None else torch.rand(1, generator=generator).item()
        force_full = rand_val < self.full_mask_probability
        if force_full:
            mask = torch.ones(answer_len, dtype=torch.bool, device=device)
            return mask, 1.0
        if self.partial_mask_min is not None and self.partial_mask_max is not None:
            low = min(self.partial_mask_min, answer_len)
            high = min(self.partial_mask_max, answer_len)
            low = max(1, low)
            high = max(low, high)
            mask_count = random.randint(low, high)
            perm = torch.randperm(answer_len, generator=generator)
            mask = torch.zeros(answer_len, dtype=torch.bool, device=device)
            mask[perm[:mask_count]] = True
            return mask, mask_count / float(answer_len)
        t_val = (
            torch.rand(1, generator=generator).item()
            if generator is not None
            else random.random()
        )
        p_val = (1.0 - self.eps) * float(t_val) + self.eps
        rand = torch.rand(answer_len, generator=generator)
        mask = rand < p_val
        if mask.sum() == 0:
            idx = torch.randint(0, answer_len, (1,), generator=generator)
            mask[idx] = True
        return mask, float(mask.sum().item()) / float(answer_len)

    def __call__(self, batch: Iterable[ClozeSample]) -> Dict[str, Any]:
        samples = list(batch)
        if not samples:
            raise ValueError("batch must not be empty.")

        batch_size = len(samples)
        device = torch.device("cpu")
        clean_ids = torch.full((batch_size, self.max_length), self.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool, device=device)
        masked_positions = torch.zeros((batch_size, self.max_length), dtype=torch.bool, device=device)
        response_mask = torch.zeros_like(masked_positions)
        answer_token_mask = torch.zeros_like(masked_positions)
        first_answer_mask = torch.zeros_like(masked_positions)
        mask_rates = torch.zeros(batch_size, dtype=torch.float32, device=device)
        masked_token_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        answer_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for idx, sample in enumerate(samples):
            prefix, answer, suffix = self._split_cloze_sentence(sample.text)
            prefix_ids, answer_ids, suffix_ids = self._encode_segments(prefix, answer, suffix)
            prefix_ids, answer_ids, suffix_ids = self._trim_segments(prefix_ids, answer_ids, suffix_ids)
            if not answer_ids:
                answer_ids = [self.mask_token_id]
            seq = prefix_ids + answer_ids + suffix_ids
            seq = seq[: self.max_length]
            seq_len = len(seq)
            clean_ids[idx, :seq_len] = torch.tensor(seq, dtype=torch.long)
            attention_mask[idx, :seq_len] = True

            answer_start = min(len(prefix_ids), self.max_length)
            answer_end = min(answer_start + len(answer_ids), self.max_length)
            if answer_end <= answer_start:
                if seq_len == 0:
                    answer_start, answer_end = 0, 0
                else:
                    answer_start = max(0, seq_len - 1)
                    answer_end = seq_len
            answer_len = max(0, answer_end - answer_start)
            if answer_len == 0:
                answer_len = 1
                answer_end = min(answer_start + 1, self.max_length)
            answer_lengths[idx] = answer_len
            start_idx = min(answer_start, self.max_length - 1)
            response_mask[idx, answer_start:answer_end] = True
            answer_token_mask[idx, answer_start:answer_end] = True
            first_answer_mask[idx, start_idx] = True

            mask, rate = self._select_mask(answer_len)
            if mask.numel() == 0:
                mask = torch.ones(answer_len, dtype=torch.bool)
                rate = 1.0
            window = answer_end - answer_start
            masked_positions[idx, answer_start:answer_end] = mask[:window]
            mask_rates[idx] = rate
            masked_token_counts[idx] = int(mask[:window].sum().item())

        input_ids = clean_ids.clone()
        target_ids = torch.full_like(clean_ids, -100)

        target_ids[masked_positions] = clean_ids[masked_positions]
        input_ids[masked_positions] = self.mask_token_id

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask.long(),
            "target_ids": target_ids,
            "masked_positions": masked_positions,
            "response_mask": response_mask,
            "answer_token_mask": answer_token_mask,
            "first_answer_mask": first_answer_mask,
            "mask_rates": mask_rates,
            "masked_token_counts": masked_token_counts,
            "answer_lengths": answer_lengths,
            "uids": [sample.uid for sample in samples],
            "answer_types": [sample.question_type for sample in samples],
            "logic_tags": [sample.logic_tags for sample in samples],
        }
        return batch_dict


def build_aux_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]", "<pad>"]})
    tokenizer.pad_token = "<pad>"
    return tokenizer


__all__ = ["ClozeDataset", "ClozeMaskCollator", "build_aux_tokenizer"]


class PromptResponseDataset(Dataset):
    """
    Minimal JSONL dataset for prompt/response QA pairs.
    """

    ANSWER_EOS_MARKERS: Tuple[str, ...] = ("<EOS>", "],")

    def __init__(
        self,
        jsonl_path: Path,
        include_context: bool = False,
        context_separator: str = "\n",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"PromptResponse dataset not found: {self.jsonl_path}")
        self.include_context = include_context
        self.context_separator = context_separator
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt", "")
                context_block = ""
                context = obj.get("context")
                if self.include_context and context:
                    if isinstance(context, list):
                        context_block = self.context_separator.join(str(x) for x in context if x).strip()
                    else:
                        context_block = str(context).strip()
                if context_block:
                    prompt_text = f"{context_block}{self.context_separator}{prompt}"
                else:
                    prompt_text = prompt
                answer_text = self._clean_answer_text(obj.get("answer", ""))
                answer_type = obj.get("answer_type")
                if not answer_type:
                    answer_type = obj.get("question_type", "name")
                if answer_type not in ("name", "job"):
                    answer_type = "job" if self._looks_like_job(prompt) else "name"
                self._records.append(
                    {
                        "prompt": prompt_text,
                        "answer": answer_text,
                        "answer_type": answer_type,
                        "logic_tags": obj.get("logic_tags", []),
                        "uid": str(obj.get("uid", len(self._records))),
                    }
                )
        if not self._records:
            raise ValueError(f"No prompt/response samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._records[idx]

    @classmethod
    def _clean_answer_text(cls, answer: Any) -> str:
        text = str(answer or "")
        cleaned = text
        for marker in cls.ANSWER_EOS_MARKERS:
            if marker and cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].rstrip()
        return cleaned

    @staticmethod
    def _looks_like_job(prompt_text: str) -> bool:
        lower = str(prompt_text or "").lower()
        return any(
            key in lower for key in ("occupation", "work as", "job", "employed as")
        )


class SFTMaskCollator:
    """Builds prompt + answer sequences (with optional EOS) and applies LLaDA-style random masking over the answer region."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_token: str = "[MASK]",
        prepend_bos: bool = False,
        add_prompt_eos: bool = True,
        append_response_eos: bool = True,
        masking_mode: str = "random",
        fixed_mask_option: Optional[Any] = None,
        eps: float = 1e-3,
        pad_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        answer_block_size: Optional[int] = None,
        randomize_answer_start: bool = False,
        user_token: str = "<start_id>user<end_id>",
        assistant_token: str = "<start_id>assistant<end_id>",
        eot_token: str = "<eot_id>",
        answer_mask_scalar: float = 1.0,
        answer_eos_mask_scalar: float = 1.0,
        mask_t_distribution: str = "uniform",
        mask_t_beta_k: float = 5.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.prepend_bos = prepend_bos
        self.add_prompt_eos = add_prompt_eos
        self.append_response_eos = append_response_eos
        self.masking_mode = "random"
        self.fixed_mask_option = None
        self.eps = float(eps)
        if not (0.0 <= self.eps < 1.0):
            raise ValueError("eps must lie within [0, 1).")
        self.generator = generator
        block_size = 8 if answer_block_size is None else answer_block_size
        self.answer_block_size = int(block_size) if block_size is not None else None

        mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        if mask_id is None or mask_id < 0:
            raise ValueError(f"Tokenizer missing mask token {mask_token}")
        self.mask_token_id = int(mask_id)
        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif tokenizer.pad_token_id is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        elif tokenizer.eos_token_id is not None:
            self.pad_token_id = int(tokenizer.eos_token_id)
        else:
            self.pad_token_id = 0
        self.bos_token_id = tokenizer.convert_tokens_to_ids("<BOS>") if prepend_bos else tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.user_token_id = tokenizer.convert_tokens_to_ids(user_token)
        self.assistant_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
        self.eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
        for name, tid in [
            ("user_token", self.user_token_id),
            ("assistant_token", self.assistant_token_id),
            ("eot_token", self.eot_token_id),
        ]:
            if tid is None or tid < 0:
                raise ValueError(f"Tokenizer missing special token for {name}. Please add it in config.")
        self.special_token_ids = {
            self.pad_token_id,
            self.mask_token_id,
            self.user_token_id,
            self.assistant_token_id,
        }
        if self.bos_token_id is not None:
            self.special_token_ids.add(self.bos_token_id)
        self.randomize_answer_start = bool(randomize_answer_start)
        self.answer_mask_scalar = float(answer_mask_scalar)
        self.answer_eos_mask_scalar = float(answer_eos_mask_scalar)
        self.mask_t_distribution = str(mask_t_distribution)
        self.mask_t_beta_k = float(mask_t_beta_k)
        if self.mask_t_distribution not in ("uniform", "beta"):
            raise ValueError(f"Unsupported mask_t_distribution: {self.mask_t_distribution}")
        if self.mask_t_distribution == "beta" and self.mask_t_beta_k <= 0:
            raise ValueError("mask_t_beta_k must be positive for beta distribution.")

    def _maybe_add_specials(self, ids: List[int], add_bos: bool, add_eos: bool) -> List[int]:
        new_ids = list(ids)
        if add_bos and self.bos_token_id is not None:
            new_ids = [self.bos_token_id] + new_ids
        if add_eos and self.eos_token_id is not None:
            new_ids = new_ids + [self.eos_token_id]
        return new_ids

    def _forward_process(
        self,
        clean_input: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        tokenwise_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return apply_random_mask(
            clean_input=clean_input,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            mask_token_id=self.mask_token_id,
            eps=self.eps,
            generator=self.generator,
            tokenwise_p_scale=tokenwise_scale,
            t_distribution=self.mask_t_distribution,
            beta_k=self.mask_t_beta_k,
        )

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tokenizer = self.tokenizer

        def _strip_special_prompt(text: str) -> str:
            for tok in ("<BOS>", "<start_id>user<end_id>", "<start_id>assistant<end_id>", "<eot_id>"):
                text = text.replace(tok, "")
            return text.strip()

        sequences: List[List[int]] = []
        prompt_lens: List[int] = []
        uids: List[str] = []
        answer_types: List[str] = []
        logic_tags: List[Sequence[str]] = []

        for idx, sample in enumerate(batch):
            prompt_text = _strip_special_prompt(str(sample["prompt"]))
            answer_text = str(sample["answer"])
            prompt_ids = []
            if self.prepend_bos and self.bos_token_id is not None:
                prompt_ids.append(self.bos_token_id)
            prompt_ids.append(self.user_token_id)
            prompt_ids += tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids.append(self.eot_token_id)
            prompt_ids.append(self.assistant_token_id)
            if self.add_prompt_eos and self.eos_token_id is not None:
                prompt_ids.append(self.eos_token_id)

            normalized_answer = answer_text
            if not normalized_answer.startswith(" "):
                normalized_answer = " " + normalized_answer
            answer_tokens = tokenizer.encode(
                normalized_answer,
                add_special_tokens=False,
            )
            if self.append_response_eos and self.eos_token_id is not None:
                answer_tokens = answer_tokens + [self.eos_token_id]
            if self.answer_block_size is not None:
                answer_tokens = answer_tokens[: self.answer_block_size]

            seq = prompt_ids + answer_tokens
            if len(seq) > self.max_length:
                overflow = len(seq) - self.max_length
                if overflow > 0 and len(prompt_ids) > 0:
                    trim = min(len(prompt_ids), overflow)
                    prompt_ids = prompt_ids[trim:]
                seq = (prompt_ids + answer_tokens)[-self.max_length:]
            sequences.append(seq)
            prompt_lens.append(min(len(prompt_ids), self.max_length))
            uids.append(str(sample.get("uid", idx)))
            sample_type = str(sample.get("answer_type", "name"))
            if sample_type not in ("name", "job"):
                sample_type = "job" if self._looks_like_job(sample.get("prompt", "")) else "name"
            answer_types.append(sample_type)
            tags = sample.get("logic_tags", [])
            logic_tags.append(tags if isinstance(tags, list) else [tags])

        batch_size = len(sequences)
        device = torch.device("cpu")
        batch_max_len = min(self.max_length, max(len(s) for s in sequences))
        clean_input = torch.full((batch_size, self.max_length), self.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool, device=device)
        prompt_lengths = torch.tensor(prompt_lens, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            seq = seq[:batch_max_len]
            fill_len = len(seq)
            if fill_len > 0:
                clean_input[i, :fill_len] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :fill_len] = True
            if fill_len < batch_max_len:
                eos_val = self.eot_token_id if self.eot_token_id is not None else self.pad_token_id
                clean_input[i, fill_len:batch_max_len] = eos_val
                attention_mask[i, fill_len:batch_max_len] = True

        answer_lengths = torch.clamp(batch_max_len - prompt_lengths, min=0)

        seq_len = attention_mask.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_mask = position_ids < prompt_lengths.unsqueeze(1)
        special_mask = torch.zeros_like(clean_input, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_mask |= clean_input.eq(token_id)
        valid_positions = attention_mask & (~prompt_mask) & (~special_mask)

        tokenwise_scale: Optional[torch.Tensor] = None
        if not math.isclose(self.answer_mask_scalar, 1.0) or not math.isclose(self.answer_eos_mask_scalar, 1.0):
            tokenwise_scale = torch.ones_like(clean_input, dtype=torch.float32, device=device)
            answer_region = attention_mask & (~prompt_mask)
            content_mask = answer_region.clone()
            eos_mask = torch.zeros_like(answer_region)
            if self.eos_token_id is not None and self.eos_token_id >= 0:
                eos_mask = answer_region & clean_input.eq(self.eos_token_id)
                content_mask = content_mask & (~clean_input.eq(self.eos_token_id))
            content_mask = content_mask & valid_positions
            eos_mask = eos_mask & valid_positions
            if not math.isclose(self.answer_mask_scalar, 1.0):
                tokenwise_scale[content_mask] = self.answer_mask_scalar
            if not math.isclose(self.answer_eos_mask_scalar, 1.0):
                tokenwise_scale[eos_mask] = self.answer_eos_mask_scalar

        noisy_input, target_ids, masked_positions, p_scalar = self._forward_process(
            clean_input,
            attention_mask,
            valid_positions,
            tokenwise_scale=tokenwise_scale,
        )
        masked_token_counts = masked_positions.sum(dim=1).to(torch.long)

        return {
            "input_ids": noisy_input,
            "attention_mask": attention_mask.long(),
            "target_ids": target_ids,
            "masked_positions": masked_positions,
            "masked_token_counts": masked_token_counts,
            "answer_lengths": answer_lengths,
            "prompt_lengths": prompt_lengths,
            "clean_input_ids": clean_input.clone(),
            "uids": uids,
            "answer_types": answer_types,
            "logic_tags": logic_tags,
            "p_mask_scalar": p_scalar,
        }

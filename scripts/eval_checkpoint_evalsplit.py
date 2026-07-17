#!/usr/bin/env python3
"""
Evaluate a single SFT checkpoint on the 10 JSONL splits prepared under
datasets/1000+500x10/qa using the same collator logic as training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets import PromptResponseDataset, SFTMaskCollator  # noqa: E402
from scripts.train_utils import build_tokenizer, load_yaml, build_model, load_checkpoint  # noqa: E402
from scripts.eval_utils import evaluate_em_token  # noqa: E402


EVAL_SPLITS: Tuple[Tuple[str, str, bool], ...] = (
    ("people_reverse", "qa_people_reverse.jsonl", False),
    ("people_forward", "qa_people_forward.jsonl", False),
    ("job", "qa_job.jsonl", False),
    ("two_hop", "qa_two_hop.jsonl", False),
    ("inversion", "qa_inversion.jsonl", False),
    ("symmetry", "qa_symmetry.jsonl", False),
    ("incontext_inversion_with_ft", "incontext_inversion_with_ft.jsonl", True),
    ("incontext_inversion_without_ft", "incontext_inversion_without_ft.jsonl", False),
    ("incontext_symmetry_with_ft", "incontext_symmetry_with_ft.jsonl", True),
    ("incontext_symmetry_without_ft", "incontext_symmetry_without_ft.jsonl", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single checkpoint on QA JSONL files."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/sft_mdm.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint-stepXXXXX.pt (model_state).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("datasets/1000+500x10/qa"),
        help="Directory containing the 10 evaluation JSONL files.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump metrics JSON.",
    )
    parser.add_argument(
        "--mask-option",
        type=str,
        default=None,
        help="Override for the fixed mask option (default uses the config's eval settings).",
    )
    return parser.parse_args()


def resolve_eval_mask_settings(
    collator_cfg: Dict[str, object],
    override_mask_option: Optional[str],
) -> Tuple[str, Optional[str]]:
    eval_mask_mode = str(
        collator_cfg.get("eval_mask_mode", "full_block") or "full_block"
    )
    eval_fixed_option = collator_cfg.get("eval_fixed_mask_option", "full")
    mask_option = override_mask_option or eval_fixed_option
    if eval_mask_mode == "fixed_window" and override_mask_option is None:
        mask_option = "full"
    return eval_mask_mode, mask_option


def build_eval_loader(
    jsonl_path: Path,
    include_context: bool,
    tokenizer,
    collator_cfg: Dict[str, object],
    max_length: int,
    batch_size: int,
    mask_mode: str,
    mask_option: Optional[str],
    num_workers: int,
    seed: int,
    context_separator: str,
) -> DataLoader:
    dataset = PromptResponseDataset(
        jsonl_path,
        include_context=include_context,
        context_separator=context_separator,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator = SFTMaskCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        mask_token=collator_cfg.get("mask_token", "[MASK]"),
        prepend_bos=collator_cfg.get("prepend_bos", False),
        add_prompt_eos=collator_cfg.get("add_prompt_eos", True),
        append_response_eos=collator_cfg.get("append_response_eos", True),
        masking_mode=mask_mode,
        fixed_mask_option=mask_option,
        eps=float(collator_cfg.get("eps", 1e-3)),
        pad_token_id=tokenizer.pad_token_id,
        generator=generator,
        answer_block_size=(
            None
            if collator_cfg.get("answer_block_size", None) in (None, "null", "None")
            else int(
                collator_cfg.get(
                    "answer_block_size",
                    collator_cfg.get("eval_block_size", 8),
                )
            )
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
    )
    return loader


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    tokenizer = build_tokenizer(config["tokenizer"])
    dataset_cfgs = config.get("datasets", {})
    dataset_key = config.get("dataset_key")
    if dataset_key and dataset_key in dataset_cfgs:
        collator_cfg = dataset_cfgs[dataset_key].get("collator", config["collator"])
        data_cfg = dataset_cfgs[dataset_key].get("data", config.get("data", {}))
    else:
        collator_cfg = config["collator"]
        data_cfg = config.get("data", {})
    max_length = int(config["tokenizer"].get("max_length", 128))
    model = build_model(config["model"], len(tokenizer), max_length)
    device = torch.device(args.device)
    load_checkpoint(args.checkpoint, model)
    model.to(device)

    default_seq_len = int(
        collator_cfg.get(
            "max_length",
            config["tokenizer"].get("max_length", 128),
        )
    )
    mask_mode, mask_option = resolve_eval_mask_settings(collator_cfg, args.mask_option)
    context_separator = str(data_cfg.get("context_separator", ""))
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)

    metrics: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    total_samples = 0
    total_token_correct = 0
    total_token_count = 0

    for offset, (name, filename, include_context) in enumerate(EVAL_SPLITS):
        jsonl_path = args.eval_dir / filename
        if not jsonl_path.exists():
            continue

        loader = build_eval_loader(
            jsonl_path,
            include_context,
            tokenizer,
            collator_cfg,
            default_seq_len,
            batch_size,
            mask_mode,
            mask_option,
            num_workers,
            seed=42 + offset,
            context_separator=context_separator,
        )

        em_rate, correct, count, token_acc, token_correct, token_count = (
            evaluate_em_token(
                model,
                loader,
                device,
                tokenizer,
                collator_cfg,
            )
        )
        metrics[name] = {
            "em": em_rate,
            "correct": correct,
            "count": count,
            "token_acc": token_acc,
            "token_correct": token_correct,
            "token_count": token_count,
        }
        total_correct += correct
        total_samples += count
        total_token_correct += token_correct
        total_token_count += token_count
        print(
            f"{name:35s} | em={em_rate * 100:6.2f}% ({correct}/{count}) "
            f"token_acc={token_acc * 100:6.2f}%"
        )

    overall_em = (
        float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
    )
    overall_token_acc = (
        total_token_correct / total_token_count if total_token_count > 0 else 0.0
    )
    metrics["qa_completion_all"] = {
        "em": overall_em,
        "correct": total_correct,
        "count": total_samples,
        "token_acc": overall_token_acc,
        "token_correct": total_token_correct,
        "token_count": total_token_count,
    }
    print("-" * 60)
    print(
        f"QA completion (all): em={overall_em * 100:6.2f}% ({total_correct}/{total_samples}) "
        f"token_acc={overall_token_acc * 100:6.2f}%"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f_out:
            json.dump(metrics, f_out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

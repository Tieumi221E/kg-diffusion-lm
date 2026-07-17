#!/usr/bin/env python3
"""
Scan a directory of SFT checkpoints and evaluate each on all QA splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_utils import build_tokenizer, load_yaml, build_model, load_checkpoint  # noqa: E402
from scripts.eval_utils import evaluate_em_token  # noqa: E402
from scripts.eval_checkpoint_evalsplit import (  # noqa: E402
    EVAL_SPLITS,
    build_eval_loader,
    resolve_eval_mask_settings,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft_mdm.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    tokenizer = build_tokenizer(config["tokenizer"])
    dataset_key = config.get("dataset_key")
    dataset_cfgs = config.get("datasets", {})
    if dataset_key and dataset_key in dataset_cfgs:
        collator_cfg = dataset_cfgs[dataset_key].get("collator", config["collator"])
        data_cfg = dataset_cfgs[dataset_key].get("data", config.get("data", {}))
    else:
        collator_cfg = config["collator"]
        data_cfg = config.get("data", {})

    max_length = int(config["tokenizer"].get("max_length", 128))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(args.checkpoint_dir.glob("checkpoint-step*.pt"))
    all_results = {}

    model = build_model(config["model"], len(tokenizer), max_length)
    model.to(device)

    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.replace("checkpoint-step", ""))
        print(f"Evaluating checkpoint at step {step}...")

        load_checkpoint(ckpt_path, model)

        mask_mode, mask_option = resolve_eval_mask_settings(collator_cfg, None)
        context_sep = str(data_cfg.get("context_separator", ""))

        ckpt_metrics = {}
        for offset, (name, filename, inc_context) in enumerate(EVAL_SPLITS):
            jsonl_path = args.eval_dir / filename
            if not jsonl_path.exists():
                continue

            loader = build_eval_loader(
                jsonl_path,
                inc_context,
                tokenizer,
                collator_cfg,
                max_length,
                128,
                mask_mode,
                mask_option,
                0,
                42 + offset,
                context_sep,
            )
            em, _, _, _, _, _ = evaluate_em_token(
                model, loader, device, tokenizer, collator_cfg
            )
            ckpt_metrics[name] = em

        all_results[step] = ckpt_metrics

    if args.output_json:
        with args.output_json.open("w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

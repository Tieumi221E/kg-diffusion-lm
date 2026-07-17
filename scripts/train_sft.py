#!/usr/bin/env python3
"""Supervised fine-tuning loop for prompt-response QA pairs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.train_utils import (  # noqa: E402
    set_global_seed,
    load_yaml,
    setup_logging,
    autocast_context,
    linear_warmup_cosine_decay_schedule,
    ensure_dir,
    save_checkpoint,
    load_checkpoint,
    cleanup_checkpoints,
    build_tokenizer,
    build_model,
    create_grad_scaler,
    mdm_loss,
)

from scripts.datasets import PromptResponseDataset, SFTMaskCollator  # noqa: E402
from scripts.eval_utils import evaluate_loss  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_dataloader(
    jsonl_path: Path,
    tokenizer: Any,
    collator_cfg: Dict[str, Any],
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    dataset = PromptResponseDataset(jsonl_path)
    collator = SFTMaskCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        mask_token=collator_cfg.get("mask_token", "[MASK]"),
        eps=float(collator_cfg.get("eps", 1e-3)),
        pad_token_id=tokenizer.pad_token_id,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sft_mdm.yaml"))
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_key = args.dataset or config.get("dataset_key")
    dataset_cfg = config["datasets"][dataset_key] if dataset_key else config

    data_cfg = dataset_cfg.get("data", config.get("data"))
    train_cfg_raw = dataset_cfg.get("training", config.get("training"))
    optimizer_cfg = dataset_cfg.get("optimizer", config.get("optimizer"))
    scheduler_cfg = dataset_cfg.get("scheduler", config.get("scheduler"))
    collator_cfg = dataset_cfg.get("collator", config.get("collator"))

    output_dir = Path(config["logging"]["output_dir"]) / dataset_key
    ensure_dir(output_dir)
    logger = setup_logging(output_dir, "train_sft")

    set_global_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config["tokenizer"])
    max_length = int(config["tokenizer"].get("max_length", 128))

    train_loader = build_dataloader(
        Path(data_cfg["train_jsonl"]),
        tokenizer,
        collator_cfg,
        int(data_cfg["batch_size"]),
        True,
        max_length,
    )
    eval_loader = build_dataloader(
        Path(data_cfg["eval_jsonl"]),
        tokenizer,
        collator_cfg,
        int(data_cfg["eval_batch_size"]),
        False,
        max_length,
    )

    model = build_model(config["model"], len(tokenizer), max_length).to(device)

    if args.pretrained_checkpoint:
        load_checkpoint(args.pretrained_checkpoint, model)
        logger.info(f"Loaded pretrained backbone from {args.pretrained_checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_cfg["lr"]))
    scheduler = linear_warmup_cosine_decay_schedule(
        optimizer, int(scheduler_cfg["warmup_steps"]), int(scheduler_cfg["total_steps"])
    )
    scaler = create_grad_scaler(device.type, True)

    start_step = 0
    step = start_step
    grad_accum = int(train_cfg_raw.get("gradient_accumulation_steps", 1))

    train_iterator = iter(train_loader)
    while step < int(train_cfg_raw["max_steps"]):
        model.train()
        optimizer.zero_grad()
        for _ in range(grad_accum):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids, att_mask = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            target_ids, p_scalar = (
                batch["target_ids"].to(device),
                batch["p_mask_scalar"].to(device),
            )

            with autocast_context(device.type, True):
                logits = model(input_ids, att_mask)
                loss = mdm_loss(logits, target_ids, p_scalar) / grad_accum
            scaler.scale(loss).backward()

        clip_norm = float(train_cfg_raw.get("clip_grad_norm", 0) or 0)
        if clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        step += 1

        if step % int(train_cfg_raw["log_every"]) == 0:
            logger.info(f"[Step {step:06d}] loss={loss.item() * grad_accum:.4f}")

        if step % int(train_cfg_raw["eval_every"]) == 0:
            eval_loss = evaluate_loss(model, eval_loader, device)
            logger.info(f"[Eval {step:06d}] loss={eval_loss:.4f}")
            model.train()

        if step % int(train_cfg_raw["save_every"]) == 0:
            save_checkpoint(
                output_dir / f"checkpoint-step{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                config,
            )
            cleanup_checkpoints(
                output_dir, int(config["logging"].get("keep_last_n_checkpoints", 1))
            )


if __name__ == "__main__":
    main()

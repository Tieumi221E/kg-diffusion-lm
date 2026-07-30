#!/usr/bin/env python3
"""Pre-training loop for the masked diffusion language model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
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
    mdm_loss_sum,
)

from scripts.datasets import (  # noqa: E402
    FixedMaskPretrainEvalDataset,
    PretrainMaskCollator,
    SyntheticLogicPretrainDataset,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_dataloaders(
    data_cfg: Dict[str, Any],
    tokenizer: Any,
    collator_cfg: Dict[str, Any],
    max_length: int,
    num_workers_override: Optional[int] = None,
) -> Tuple[DataLoader, Dict[str, DataLoader], Dict[str, str]]:
    train_dataset = SyntheticLogicPretrainDataset(
        data_path=Path(data_cfg["train_jsonl"])
    )
    eval_jsonl = data_cfg.get("eval_jsonl")
    fixed_eval_jsonl = data_cfg.get("fixed_eval_jsonl")
    fixed_eval_multi = data_cfg.get("fixed_eval_jsonl_multi")

    collator = PretrainMaskCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        mask_token=collator_cfg.get("mask_token", "[MASK]"),
        random_crop_prob=collator_cfg.get("random_crop_prob", 0.01),
        eps=collator_cfg.get("eps", 1e-3),
        prepend_bos=collator_cfg.get("prepend_bos", False),
        append_eos=collator_cfg.get("append_eos", False),
        pad_token_id=tokenizer.pad_token_id,
    )

    num_workers = (
        int(num_workers_override)
        if num_workers_override is not None
        else int(data_cfg.get("num_workers", 0))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    eval_loaders: Dict[str, DataLoader] = {}
    eval_modes: Dict[str, str] = {}

    if fixed_eval_multi:
        for idx, path_str in enumerate(fixed_eval_multi):
            path = Path(path_str)
            if not path.exists():
                continue
            ds = FixedMaskPretrainEvalDataset(path)
            loader = DataLoader(
                ds,
                batch_size=int(data_cfg.get("eval_batch_size", 64)),
                shuffle=False,
                collate_fn=None,
            )
            eval_loaders[f"fixed_eval_{idx}"] = loader
            eval_modes[f"fixed_eval_{idx}"] = "token_acc"
    elif fixed_eval_jsonl:
        ds = FixedMaskPretrainEvalDataset(Path(fixed_eval_jsonl))
        eval_loaders["fixed_eval"] = DataLoader(
            ds, batch_size=int(data_cfg.get("eval_batch_size", 64)), shuffle=False
        )
        eval_modes["fixed_eval"] = "token_acc"
    elif eval_jsonl:
        ds = SyntheticLogicPretrainDataset(data_path=Path(eval_jsonl))
        eval_loaders["loss_eval"] = DataLoader(
            ds,
            batch_size=int(data_cfg.get("eval_batch_size", 64)),
            shuffle=False,
            collate_fn=collator,
        )
        eval_modes["loss_eval"] = "loss"

    return train_loader, eval_loaders, eval_modes


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids, att_mask = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            target_ids, p_scalar = (
                batch["target_ids"].to(device),
                batch["p_mask_scalar"].to(device),
            )
            logits = model(input_ids, att_mask)
            l_sum, c = mdm_loss_sum(logits, target_ids, p_scalar)
            total_loss += l_sum.item()
            total_count += c.item()
    return total_loss / max(1.0, total_count)


def evaluate_acc(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_correct, total_masked, total_em, total_samples = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids, att_mask = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            target_ids = batch["target_ids"].to(device)
            logits = model(input_ids, att_mask)
            preds = logits.argmax(dim=-1)
            mask = target_ids != -100
            correct = (preds == target_ids) & mask
            total_correct += correct.sum().item()
            total_masked += mask.sum().item()

            per_sample_masked = mask.sum(dim=1)
            if per_sample_masked.numel() > 0:
                per_sample_correct = correct.sum(dim=1)
                total_em += (per_sample_correct == per_sample_masked).sum().item()
                total_samples += per_sample_masked.numel()
    return {
        "token_acc": total_correct / max(1, total_masked),
        "em": total_em / max(1, total_samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_mdm.yaml"))
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--resume", type=Path)
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
    logger = setup_logging(output_dir, "train_mdm")

    set_global_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config["tokenizer"])
    max_length = int(config["tokenizer"].get("max_length", 128))

    train_loader, eval_loaders, eval_modes = build_dataloaders(
        data_cfg, tokenizer, collator_cfg, max_length
    )

    model = build_model(config["model"], len(tokenizer), max_length).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimizer_cfg["lr"]))
    scheduler = linear_warmup_cosine_decay_schedule(
        optimizer, int(scheduler_cfg["warmup_steps"]), int(scheduler_cfg["total_steps"])
    )
    scaler = create_grad_scaler(device.type, True)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

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
            for name, loader in eval_loaders.items():
                if eval_modes[name] == "token_acc":
                    m = evaluate_acc(model, loader, device)
                    logger.info(
                        f"[Eval {name}] token_acc={m['token_acc']:.4f} em={m['em']:.4f}"
                    )
                else:
                    eval_loss = evaluate_loss(model, loader, device)
                    logger.info(f"[Eval {name}] loss={eval_loss:.4f}")

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
                output_dir, int(config["logging"].get("keep_last_n_checkpoints", 3))
            )


if __name__ == "__main__":
    main()

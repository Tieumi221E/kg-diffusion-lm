import math
import os
import random
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Iterable
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2TokenizerFast

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_logging(output_dir: Path, name: str, filename: str = "train.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(output_dir / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

@contextmanager
def autocast_context(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            with torch.amp.autocast(device_type=device_type, enabled=enabled):
                yield
                return
        except TypeError:
            with torch.amp.autocast(enabled=enabled):
                yield
                return
    with torch.cuda.amp.autocast(enabled=enabled):
        yield

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def linear_warmup_cosine_decay_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    config: Dict[str, Any],
) -> None:
    model_to_save = unwrap_model(model)
    payload = {
        "step": step,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": config,
    }
    torch.save(payload, path)

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> int:
    payload = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    return int(payload.get("step", 0))

def cleanup_checkpoints(directory: Path, keep_last: int, keep_best: bool = True) -> None:
    if keep_last is None or keep_last <= 0:
        return
    ckpts = sorted(directory.glob("checkpoint-step*.pt"), key=os.path.getmtime)
    if len(ckpts) <= keep_last:
        return
    to_remove = ckpts[: len(ckpts) - keep_last]
    for path in to_remove:
        path.unlink(missing_ok=True)
    if keep_best:
        best_path = directory / "checkpoint-best.pt"
        if best_path.exists():
            os.utime(best_path, None)

def build_tokenizer(tokenizer_cfg: Dict[str, Any]) -> GPT2TokenizerFast:
    name = tokenizer_cfg["name_or_path"]
    tokenizer = GPT2TokenizerFast.from_pretrained(name)
    add_tokens = tokenizer_cfg.get("add_special_tokens", {})
    if add_tokens:
        tokenizer.add_special_tokens(add_tokens)
    pad_token = tokenizer_cfg.get("pad_token")
    if pad_token:
        tokenizer.pad_token = pad_token
    return tokenizer

def build_model(model_cfg: Dict[str, Any], vocab_size: int, max_length: int) -> "DiffusionTransformer":
    from diffusion_core.model import DiffusionTransformer
    return DiffusionTransformer(
        vocab_size=vocab_size,
        max_position_embeddings=max_length,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_hidden_layers"]),
        num_heads=int(model_cfg["num_attention_heads"]),
        intermediate_size=int(model_cfg["intermediate_size"]),
        emb_dropout=float(model_cfg.get("emb_dropout", 0.0)),
        resid_dropout=float(model_cfg.get("resid_dropout", 0.0)),
        attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
    )

def create_grad_scaler(device_type: str, enabled: bool) -> "torch.cuda.amp.GradScaler":
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        GradScalerCls = torch.amp.GradScaler  # type: ignore[attr-defined]
        try:
            return GradScalerCls(device_type=device_type, enabled=enabled)  # type: ignore[call-arg]
        except TypeError:
            return GradScalerCls(enabled=enabled)
    return torch.amp.GradScaler('cuda', enabled=enabled)

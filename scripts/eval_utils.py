import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Dict, Tuple
from diffusion_core.loss import diffusion_loss_sum


def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids, att_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            target_ids, p_scalar = batch["target_ids"].to(device), batch["p_mask_scalar"].to(device)
            logits = model(input_ids, att_mask)
            l_sum, c = diffusion_loss_sum(logits, target_ids, p_scalar)
            total_loss += l_sum.item()
            total_count += c.item()
    return total_loss / max(1.0, total_count)


def evaluate_em_token(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokenizer: Any,
    sampler_cfg: Dict[str, Any],
) -> Tuple[float, int, int, float, int, int]:
    """
    Single-pass greedy evaluation over masked answer positions.
    Returns (em_rate, correct, total, token_acc, token_correct, token_total).
    """
    model.eval()
    total_correct, total_samples = 0, 0
    total_token_correct, total_token_count = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            att_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            masked_positions = batch["masked_positions"].to(device)

            logits = model(input_ids, att_mask)
            preds = logits.argmax(dim=-1)

            valid = masked_positions & (target_ids != -100)
            correct = (preds == target_ids) & valid

            total_token_correct += int(correct.sum().item())
            total_token_count += int(valid.sum().item())

            per_valid = valid.sum(dim=1)
            per_correct = correct.sum(dim=1)
            em = (per_correct == per_valid) & (per_valid > 0)
            total_correct += int(em.sum().item())
            total_samples += int((per_valid > 0).sum().item())

    em_rate = total_correct / max(1, total_samples)
    token_acc = total_token_correct / max(1, total_token_count)
    return em_rate, total_correct, total_samples, token_acc, total_token_correct, total_token_count

import torch
import random
from typing import List, Optional, Tuple, Any

class DiffusionSampler:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        mask_token_id: int,
        device: torch.device,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id

    def _sample(self, logits: torch.Tensor) -> int:
        logits = logits / max(self.temperature, 1e-5)
        if self.top_k is not None and 0 < self.top_k < logits.numel():
            values, indices = torch.topk(logits, self.top_k)
            probs = torch.softmax(values, dim=-1)
            pick = torch.multinomial(probs, num_samples=1).item()
            return int(indices[pick].item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 8,
        block_size: int = 8,
        steps_per_block: int = 8,
        remask_mode: str = "random",
    ) -> List[int]:
        """
        Iterative denoising generation.
        """
        self.model.eval()
        generated = []
        remaining = max_new_tokens
        
        while remaining > 0:
            slots = min(block_size, remaining)
            block_out, eos_hit = self._generate_block(
                prompt_ids + generated, slots, steps_per_block, remask_mode
            )
            generated.extend(block_out)
            remaining -= len(block_out)
            if eos_hit or len(block_out) < slots:
                break
        return generated

    def _generate_block(
        self,
        current_seq: List[int],
        slots: int,
        steps: int,
        remask_mode: str,
    ) -> Tuple[List[int], bool]:
        block_tokens = [self.mask_token_id] * slots
        token_confidence = [0.0] * slots
        
        for step_idx in range(steps):
            full_seq = current_seq + block_tokens
            input_ids = torch.tensor([full_seq], device=self.device)
            att_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                logits = self.model(input_ids, att_mask)[0] # [Seq, Vocab]
            
            for i in range(slots):
                if block_tokens[i] != self.mask_token_id:
                    continue
                pos = len(current_seq) + i
                if pos >= logits.size(0): break
                
                pos_logits = logits[pos].clone()
                # Prevent predicting MASK or PAD
                pos_logits[self.mask_token_id] = -float('inf')
                if self.pad_token_id is not None and self.pad_token_id != self.eos_token_id:
                    pos_logits[self.pad_token_id] = -float('inf')
                
                # Safety fallback: if all logits are -inf, use raw logits (minus MASK)
                if torch.isinf(pos_logits).all():
                    pos_logits = logits[pos].clone()
                    pos_logits[self.mask_token_id] = -float('inf')

                token_id = self._sample(pos_logits)
                probs = torch.softmax(pos_logits / max(self.temperature, 1e-5), dim=-1)
                token_confidence[i] = float(probs[token_id].item())
                block_tokens[i] = token_id

            if step_idx < steps - 1:
                remask_prob = (steps - step_idx - 1) / steps
                if remask_mode == "low_confidence":
                    # Sort indices by confidence
                    indices = sorted(range(slots), key=lambda i: token_confidence[i])
                    num_remask = int(round(remask_prob * slots))
                    for i in indices[:num_remask]:
                        block_tokens[i] = self.mask_token_id
                else:
                    for i in range(slots):
                        if random.random() < remask_prob:
                            block_tokens[i] = self.mask_token_id

        # Post-process block for EOS
        eos_hit = False
        if self.eos_token_id in block_tokens:
            eos_pos = block_tokens.index(self.eos_token_id)
            block_tokens = block_tokens[:eos_pos]
            eos_hit = True
            
        return block_tokens, eos_hit

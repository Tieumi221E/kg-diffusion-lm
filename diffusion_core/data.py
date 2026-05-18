import torch
from torch.utils.data import Dataset
from typing import List, Optional, Dict, Any
from .masking import apply_diffusion_mask

class SimpleTextDataset(Dataset):
    """
    A simple dataset that takes a list of pre-tokenized ID lists or strings.
    """
    def __init__(self, tokenized_texts: List[List[int]]):
        self.data = tokenized_texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DiffusionCollator:
    def __init__(
        self,
        tokenizer: Any,
        mask_token_id: int,
        max_length: int,
        eps: float = 1e-3,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.max_length = max_length
        self.eps = eps

    def __call__(self, batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_batch_len = min(self.max_length, max(len(x) for x in batch))
        input_ids_list = []
        attention_mask_list = []
        for tokens in batch:
            tokens = tokens[:self.max_length]
            pad_len = max_batch_len - len(tokens)
            input_ids_list.append(tokens + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask_list.append([1] * len(tokens) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        valid_mask = attention_mask.clone()

        noisy_input, target_ids, _, p_scalar = apply_diffusion_mask(
            clean_input=input_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            valid_mask=valid_mask,
            eps=self.eps
        )
        
        return {
            "input_ids": noisy_input,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "p_mask_scalar": p_scalar
        }

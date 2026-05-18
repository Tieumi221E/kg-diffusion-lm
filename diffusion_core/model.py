import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(norm + self.eps)
        return self.weight * x * rms

class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(self.w1(x) * F.silu(self.w2(x))))

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention_dropout: float,
        resid_dropout: float,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(resid_dropout)
        self.ff = SwiGLUFeedForward(hidden_size, intermediate_size, resid_dropout)
        self.ff_dropout = nn.Dropout(resid_dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: (batch, seq) bool with True for valid tokens
        key_padding_mask = ~attention_mask.bool()
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.attn_dropout(attn_output)

        residual = x
        x_norm = self.norm2(x)
        ff_output = self.ff(x_norm)
        x = residual + self.ff_dropout(ff_output)
        return x

class DiffusionTransformer(nn.Module):
    """
    A standalone Bidirectional Transformer for Masked Diffusion.
    """
    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        emb_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_position_embeddings, hidden_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                attention_dropout,
                resid_dropout,
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight
        self.max_pos = max_position_embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        # Support variable sequence lengths correctly
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.head(x)

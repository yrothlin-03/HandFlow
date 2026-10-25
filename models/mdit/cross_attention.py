import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        out, _ = self.attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=(
                ~attention_mask.bool() if attention_mask is not None else None
            ),
            need_weights=False,
        )
        return out

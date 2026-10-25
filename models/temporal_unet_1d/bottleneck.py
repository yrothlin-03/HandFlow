import torch.nn as nn
from .modules import FiLMResBlock
import torch

class MHA(nn.Module):
    def __init__(self, channels, num_heads=8, dropout=0.0, sequence_length=15):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.norm = nn.LayerNorm(channels)
        self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, channels))
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, frame_mask=None):
        tokens = x.transpose(1, 2)

        tokens = tokens + self.positional_embedding[:, :tokens.shape[1]]

        normalized = self.norm(tokens)

        padding_mask = None
        if frame_mask is not None:
            padding_mask = ~frame_mask.bool()

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return (tokens + attended).transpose(1, 2)


class BottleNeck(nn.Module):
    def __init__(
        self,
        channels,
        cond_dim,
        attention_heads=8,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.block_in = FiLMResBlock(channels, channels, cond_dim)
        self.attention = MHA(
            channels,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.block_out = FiLMResBlock(channels, channels, cond_dim)

    def forward(self, x, cond, frame_mask=None):
        x = self.block_in(x, cond)
        x = self.attention(x, frame_mask)
        return self.block_out(x, cond)

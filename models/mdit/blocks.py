import torch
import torch.nn as nn

from .adaln import AdaLNZero, modulate
from .cross_attention import CrossAttention


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )
        self.norm2 = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )
        self.norm3 = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )

        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )

        self.cross_attn = CrossAttention(
            hidden_dim,
            num_heads,
        )

        mlp_dim = int(hidden_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

        self.adaln = AdaLNZero(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        text: torch.Tensor,
        attention_mask: torch.Tensor = None,
        frame_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaln(c)

        h = modulate(self.norm1(x), shift_sa, scale_sa)

        h, _ = self.self_attn(
            h, h, h,
            key_padding_mask=(
                ~frame_mask.bool() if frame_mask is not None else None
            ),
            need_weights=False,
        )

        x = x + gate_sa[:, None, :] * h

        h = modulate(self.norm2(x), shift_ca, scale_ca)

        h = self.cross_attn(
            h,
            context=text,
            attention_mask=attention_mask,
        )

        x = x + gate_ca[:, None, :] * h

        h = modulate(self.norm3(x), shift_mlp, scale_mlp)

        h = self.mlp(h)

        x = x + gate_mlp[:, None, :] * h

        return x

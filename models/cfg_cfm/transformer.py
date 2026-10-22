
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding
from .timestep_emb import TimestepEmbedder


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(
            B, T, 3, self.num_heads, self.head_dim
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k)

        attn_mask = None

        if mask is not None:
            attn_mask = mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context, context_mask=None):
        B, T, D = x.shape
        L = context.shape[1]

        q = self.q_proj(x).reshape(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        kv = self.kv_proj(context).reshape(
            B, L, 2, self.num_heads, self.head_dim
        )

        k, v = kv.unbind(dim=2)

        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None

        if context_mask is not None:
            attn_mask = context_mask[:, None, None, :].bool()

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)

        self.self_attn = SelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads)

        mlp_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim),
        )

        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self,
        x,
        t_emb,
        text_emb,
        text_mask=None,
        motion_mask=None,
    ):
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_ff, scale_ff, gate_ff,
        ) = self.adaLN(t_emb).chunk(9, dim=-1)

        h = modulate(self.norm1(x), shift_sa, scale_sa)

        x = x + gate_sa[:, None, :] * self.self_attn(
            h, motion_mask
        )

        h = modulate(self.norm2(x), shift_ca, scale_ca)

        x = x + gate_ca[:, None, :] * self.cross_attn(
            h, text_emb, text_mask
        )

        h = modulate(self.norm3(x), shift_ff, scale_ff)

        x = x + gate_ff[:, None, :] * self.mlp(h)

        if motion_mask is not None:
            x = x * motion_mask[..., None]

        return x


class TemporalDiT(nn.Module):
    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=False
        )

        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x,
        t,
        text_emb,
        text_mask=None,
        motion_mask=None,
    ):
        h = self.input_proj(x)
        t_emb = self.time_embedder(t)

        for block in self.blocks:
            h = block(
                h,
                t_emb,
                text_emb,
                text_mask,
                motion_mask,
            )

        shift, scale = self.final_modulation(t_emb).chunk(
            2, dim=-1
        )

        h = modulate(self.final_norm(h), shift, scale)
        velocity = self.output_proj(h)

        if motion_mask is not None:
            velocity = velocity * motion_mask[..., None]

        return velocity

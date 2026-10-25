import torch
import torch.nn as nn

from .embeddings import TimestepEmbedder, TemporalEmbedding
from .conditioning import TextConditioner, InitialStateConditioner
from .blocks import DiTBlock


class MotionDiT(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        max_frames: int = 256,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.in_channels = motion_dim
        self.time_length = max_frames
        self.text_dim = text_dim
        self.motion_embed = nn.Linear(motion_dim, hidden_dim)

        self.time_embed = TimestepEmbedder(hidden_dim)
        self.pos_embed = TemporalEmbedding(max_frames, hidden_dim)

        self.text_conditioner = TextConditioner(
            text_dim, hidden_dim
        )
        self.null_text = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.init_conditioner = InitialStateConditioner(
            motion_dim, hidden_dim
        )

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, motion_dim)

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        drop_mask: torch.Tensor = None,
        frame_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        x_init: torch.Tensor = None,
    ) -> torch.Tensor:
        if x_t.ndim != 3 or x_t.shape[1] != self.in_channels:
            raise ValueError("x_t must have shape [B, C, T]")
        if x_t.shape[2] > self.time_length:
            raise ValueError("Motion length exceeds max_frames")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape [B]")
        if (
            text.ndim != 3
            or text.shape[0] != x_t.shape[0]
            or text.shape[2] != self.text_dim
        ):
            raise ValueError("text must have shape [B, L, D]")
        if (
            attention_mask is not None
            and attention_mask.shape != text.shape[:2]
        ):
            raise ValueError("attention_mask must have shape [B, L]")
        if frame_mask is not None and frame_mask.shape != (
            x_t.shape[0],
            x_t.shape[2],
        ):
            raise ValueError("frame_mask must have shape [B, T]")
        if drop_mask is not None and drop_mask.shape != (x_t.shape[0],):
            raise ValueError("drop_mask must have shape [B]")
        if x_init is not None and x_init.shape != (
            x_t.shape[0],
            self.in_channels,
        ):
            raise ValueError("x_init must have shape [B, C]")

        x = x_t.transpose(1, 2)
        x = self.motion_embed(x)
        x = self.pos_embed(x)

        if x_init is not None:
            x = self.init_conditioner(x, x_init)

        c = self.time_embed(t)
        text_cond = self.text_conditioner(text)
        if drop_mask is not None:
            drop_mask = drop_mask.bool()
            text_cond = torch.where(
                drop_mask[:, None, None],
                self.null_text.expand_as(text_cond),
                text_cond,
            )
            if attention_mask is not None:
                attention_mask = torch.where(
                    drop_mask[:, None],
                    torch.ones_like(attention_mask, dtype=torch.bool),
                    attention_mask.bool(),
                )

        for block in self.blocks:
            x = block(
                x=x,
                c=c,
                text=text_cond,
                attention_mask=attention_mask,
                frame_mask=frame_mask,
            )

        x = self.final_norm(x)
        velocity = self.output(x)

        return velocity.transpose(1, 2)

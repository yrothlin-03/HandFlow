import math
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        frequency_dim: int = 256,
        time_scale: float = 1000.0,
    ):
        super().__init__()

        self.frequency_dim = frequency_dim
        self.time_scale = time_scale

        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2

        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half, device=t.device, dtype=torch.float32
            )
            / half
        )

        angles = t.float()[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [torch.cos(angles), torch.sin(angles)],
            dim=-1,
        )

        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )

        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * self.time_scale
        emb = self.timestep_embedding(t, self.frequency_dim)
        return self.mlp(emb)


class TemporalEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_dim: int):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_frames, hidden_dim)
        )

        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:, :x.shape[1], :]
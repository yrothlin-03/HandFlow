
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

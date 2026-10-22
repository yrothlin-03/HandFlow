
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()

        assert head_dim % 2 == 0

        inv_freq = base ** (
            -torch.arange(0, head_dim, 2).float() / head_dim
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        B, H, T, D = q.shape

        positions = torch.arange(
            T, device=q.device, dtype=torch.float32
        )

        angles = torch.outer(positions, self.inv_freq)

        cos = angles.cos()[None, None]
        sin = angles.sin()[None, None]

        def rotate(x):
            even = x[..., 0::2].float()
            odd = x[..., 1::2].float()

            y_even = even * cos - odd * sin
            y_odd = even * sin + odd * cos

            return torch.stack(
                (y_even, y_odd), dim=-1
            ).flatten(-2).to(x.dtype)

        return rotate(q), rotate(k)

import torch
import torch.nn as nn


class TextConditioner(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        return self.proj(text)


class InitialStateConditioner(nn.Module):
    def __init__(self, motion_dim: int, hidden_dim: int):
        super().__init__()

        self.init_proj = nn.Linear(motion_dim, hidden_dim)
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(
        self,
        motion: torch.Tensor,
        x_init: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, num_frames, _ = motion.shape

        init = self.init_proj(x_init)
        init = init.unsqueeze(1).expand(-1, num_frames, -1)

        return self.fusion(torch.cat([motion, init], dim=-1))
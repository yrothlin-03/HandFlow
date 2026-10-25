import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 9 * hidden_dim),
        )

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(9, dim=-1)
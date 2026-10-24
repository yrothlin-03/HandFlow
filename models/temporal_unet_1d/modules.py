import torch
import torch.nn as nn


class ConditionalEmbedder(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.null_text = nn.Parameter(torch.zeros(1, cond_dim))

    def forward(self, e_time, e_text, drop_mask=None):
        if drop_mask is not None:
            e_text = torch.where(
                drop_mask[:, None],
                self.null_text.expand_as(e_text),
                e_text,
            )
        concat = torch.cat([e_time, e_text], dim=-1)
        return self.fusion(concat)


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        num_groups = min(8, out_channels)
        while out_channels % num_groups:
            num_groups -= 1

        self.norm1 = nn.GroupNorm(num_groups, out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

        self.act = nn.Mish()

        self.film_proj = nn.Linear(cond_dim, 2 * out_channels)

        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, X, e_c):
        res = self.residual(X)

        h = self.conv1(X)
        h = self.norm1(h)

        gamma, beta = self.film_proj(e_c).chunk(2, -1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        h = (1 + gamma) * h + beta

        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + res


class DownSample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, X):
        return self.conv(X)


class UpSample1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, X):
        return self.conv(X)

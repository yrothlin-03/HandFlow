import torch
import torch.nn as nn
from .modules import FiLMResBlock, ConditionalEmbedder, UpSample1D, DownSample1D
from .text_embedder import TextEmbedder
from .timestep_embedder import TimestepEmbedder
from .bottleneck import BottleNeck


class Encoder(nn.Module):
    def __init__(
        self,
        base_channels=64,
        cond_dim=128,
        n_stages=3,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = base_channels

        for _ in range(n_stages):
            self.blocks.append(FiLMResBlock(channels, channels, cond_dim))
            self.downsamples.append(DownSample1D(channels, channels * 2))
            channels *= 2

        self.out_channels = channels

    def forward(self, x, cond):
        skips = []

        for block, down in zip(self.blocks, self.downsamples):
            x = block(x, cond)
            skips.append(x)
            x = down(x)

        return x, skips


class Decoder(nn.Module):
    def __init__(
        self,
        base_channels=64,
        cond_dim=128,
        n_stages=3,
    ):
        super().__init__()

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        channels = base_channels * 2 ** n_stages

        for _ in range(n_stages):
            out_channels = channels // 2
            self.upsamples.append(UpSample1D(channels, out_channels))
            self.blocks.append(FiLMResBlock(channels, out_channels, cond_dim))
            channels = out_channels

        self.out_channels = channels

    def forward(self, x, skips, cond):
        for upsample, block, skip in zip(self.upsamples, self.blocks, reversed(skips)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)

        return x


class TemporalUnet1D(nn.Module):
    def __init__(
        self,
        time_length,
        in_channels,
        d_in,
        n_stages,
        d_cond,
    ):
        super().__init__()

        if time_length % (2 ** n_stages) != 0:
            raise ValueError("time_length must be divisible by 2 ** n_stages")

        self.time_length = time_length
        self.in_channels = in_channels
        self.in_proj = nn.Conv1d(in_channels, d_in, kernel_size=3, padding=1)
        self.time_embedder = TimestepEmbedder(d_cond)
        self.text_embedder = TextEmbedder(d_cond)
        self.conditional_embedder = ConditionalEmbedder(d_cond)

        self.encoder = Encoder(d_in, d_cond, n_stages)
        self.bottleneck = BottleNeck(self.encoder.out_channels, d_cond)
        self.decoder = Decoder(d_in, d_cond, n_stages)
        self.out_proj = nn.Conv1d(
            self.decoder.out_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, timesteps, texts, drop_mask=None):
        e_time = self.time_embedder(timesteps)
        e_text = self.text_embedder(texts)
        cond = self.conditional_embedder(e_time, e_text, drop_mask)

        x = self.in_proj(x)
        x, skips = self.encoder(x, cond)
        x = self.bottleneck(x, cond)
        x = self.decoder(x, skips, cond)
        return self.out_proj(x)

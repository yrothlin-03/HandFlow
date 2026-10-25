import torch
import torch.nn as nn
import torch.nn.functional as F
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
        blocks_per_stage=2,
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = base_channels

        for _ in range(n_stages):
            self.stages.append(
                nn.ModuleList([
                    FiLMResBlock(channels, channels, cond_dim)
                    for _ in range(blocks_per_stage)
                ])
            )
            self.downsamples.append(DownSample1D(channels, channels * 2))
            channels *= 2

        self.out_channels = channels

    def forward(self, x, cond):
        skips = []

        for stage, down in zip(self.stages, self.downsamples):
            for block in stage:
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
        blocks_per_stage=2,
    ):
        super().__init__()

        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()

        channels = base_channels * 2 ** n_stages

        for _ in range(n_stages):
            out_channels = channels // 2
            self.upsamples.append(UpSample1D(channels, out_channels))
            self.stages.append(
                nn.ModuleList([
                    FiLMResBlock(channels, out_channels, cond_dim),
                    *[
                        FiLMResBlock(out_channels, out_channels, cond_dim)
                        for _ in range(blocks_per_stage - 1)
                    ],
                ])
            )
            channels = out_channels

        self.out_channels = channels

    def forward(self, x, skips, cond):
        for upsample, stage, skip in zip(
            self.upsamples,
            self.stages,
            reversed(skips),
        ):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            for block in stage:
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
        blocks_per_stage=2,
        attention_heads=8,
        attention_dropout=0.0,
    ):
        super().__init__()

        if time_length % (2 ** n_stages) != 0:
            raise ValueError("time_length must be divisible by 2 ** n_stages")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be at least 1")

        self.time_length = time_length
        self.in_channels = in_channels
        self.in_proj = nn.Conv1d(in_channels, d_in, kernel_size=3, padding=1)
        self.time_embedder = TimestepEmbedder(d_cond)
        self.text_embedder = TextEmbedder(d_cond)
        self.conditional_embedder = ConditionalEmbedder(d_cond)

        self.encoder = Encoder(
            d_in,
            d_cond,
            n_stages,
            blocks_per_stage,
        )
        self.bottleneck = BottleNeck(
            self.encoder.out_channels,
            d_cond,
            attention_heads,
            attention_dropout,
        )
        self.decoder = Decoder(
            d_in,
            d_cond,
            n_stages,
            blocks_per_stage,
        )
        self.out_proj = nn.Conv1d(
            self.decoder.out_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x,
        timesteps,
        texts,
        drop_mask=None,
        frame_mask=None,
        attention_mask=None,
    ):
        e_time = self.time_embedder(timesteps)
        e_text = self.text_embedder(texts, attention_mask)
        cond = self.conditional_embedder(e_time, e_text, drop_mask)

        x = self.in_proj(x)
        x, skips = self.encoder(x, cond)

        bottleneck_mask = None
        if frame_mask is not None:
            bottleneck_mask = F.interpolate(
                frame_mask[:, None].float(),
                size=x.shape[-1],
                mode="nearest",
            )[:, 0].bool()

        x = self.bottleneck(x, cond, bottleneck_mask)
        x = self.decoder(x, skips, cond)
        return self.out_proj(x)

import torch.nn as nn
from .modules import FiLMResBlock


class BottleNeck(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            FiLMResBlock(channels, channels, cond_dim),
            FiLMResBlock(channels, channels, cond_dim),
        ])

    def forward(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x

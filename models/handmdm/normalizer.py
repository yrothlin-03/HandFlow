# Source: https://github.com/leorebensabath/HandMDM/blob/main/src/normalizer.py
# Adaptation: the statistics can be passed directly as tensors (mean, std), so we can
# use the HandX statistics stored in our LMDB instead of only mean.pt/std.pt files.
# Loading from base_dir and the buffer names (mean, std) are unchanged, so the
# "motion_normalizer.*" / "text_normalizer.*" checkpoint keys still load.
# save() is removed: our statistics are computed by data/handx_preprocessor.py.

import os
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        base_dir: str = None,
        eps: float = 1e-12,
        disable: bool = False,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.eps = eps

        self.disable = disable
        if not disable:
            if mean is None or std is None:
                if base_dir is None:
                    raise ValueError("Normalizer needs (mean, std) or base_dir")
                mean = torch.load(os.path.join(base_dir, "mean.pt"))
                std = torch.load(os.path.join(base_dir, "std.pt"))
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x

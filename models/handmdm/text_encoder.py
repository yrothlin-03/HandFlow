# Source:
#   CLIP_wrapper: https://github.com/leorebensabath/HandMDM/blob/main/src/model/text_encoder.py
#   to_tx_dict:   format built in https://github.com/leorebensabath/HandMDM/blob/main/src/data/text.py
#                 (one token per sentence) and src/data/collate.py (padding + mask)
# CLIP package: https://github.com/openai/CLIP (pinned by HandMDM)
#
# Adaptations:
#   - Only CLIP_wrapper is kept: Llama3_wrapper, OneHotEncoding_wrapper (hard-coded cluster
#     path), HF_wrapper and the TextToEmb dispatcher are not used by the released HandMDM.
#   - CLIP_wrapper itself is a verbatim copy.
#   - to_tx_dict is new: it gathers in one place the conversion that HandMDM spreads over
#     TextEmbeddings (CLIP sentence vector = sequence of length 1) and collate_text_motion
#     (length + mask), and returns the "tx" dict expected by GaussianDiffusion.
# The encoder is kept outside GaussianDiffusion on purpose: as a submodule its frozen
# weights would enter the state_dict and break strict loading of the checkpoints.

import torch
import torch.nn as nn

from typing import Dict, List


class CLIP_wrapper(nn.Module):
    def __init__(self, modelname: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device

        import clip

        model, preprocess = clip.load(modelname, device)
        self.tokenizer = clip.tokenize
        self.clip_model = model.eval()

        # Freeze the weights just in case
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> nn.Module:
        # override it to be always false
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def forward(self, texts: List[str], device=None) -> Dict:
        device = device if device is not None else self.device
        tokens = self.tokenizer(texts, truncate=True).to(device)
        return self.clip_model.encode_text(tokens).float()


def to_tx_dict(embeddings: torch.Tensor) -> Dict:
    # [N, 512] sentence embeddings -> sequences of one token
    x = embeddings[:, None, :]
    length = torch.ones(len(x), dtype=torch.long, device=x.device)
    mask = torch.ones(len(x), 1, dtype=torch.bool, device=x.device)
    return {"x": x, "length": length, "mask": mask}

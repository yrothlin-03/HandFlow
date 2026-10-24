
import torch.nn as nn

class TextEmbedder(nn.Module):
    def __init__(self, hidden_dim, d_embedding=512):
        super().__init__()
        self.proj = nn.Linear(d_embedding, hidden_dim)

    def forward(self, text_embedding):
        if text_embedding.ndim != 2:
            raise ValueError("text_embedding must be of shape [B, 512]")

        return self.proj(text_embedding)

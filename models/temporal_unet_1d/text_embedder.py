
import torch.nn as nn

class TextEmbedder(nn.Module):
    def __init__(self, hidden_dim, d_embedding=512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(d_embedding),
            nn.Linear(d_embedding, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, text_embedding, attention_mask):
        if text_embedding.ndim != 3:
            raise ValueError(
                "text_embedding must have shape [B, L, 512]"
            )
        if (
            attention_mask is None
            or attention_mask.shape != text_embedding.shape[:2]
        ):
            raise ValueError("attention_mask must have shape [B, L]")

        mask = attention_mask.unsqueeze(-1).to(text_embedding.dtype)
        pooled = (text_embedding * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)
        return self.proj(pooled)

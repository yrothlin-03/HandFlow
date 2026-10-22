
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class TextEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        model_name: str = "google-t5/t5-small",
        max_length: int = 64,
    ):
        super().__init__()

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.proj = nn.Linear(
            self.encoder.config.d_model,
            hidden_dim,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, texts: list[str]):
        device = self.proj.weight.device

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )

        embeddings = self.proj(outputs.last_hidden_state)

        return embeddings, tokens["attention_mask"]

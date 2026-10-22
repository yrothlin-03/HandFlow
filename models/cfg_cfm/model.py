
import torch
import torch.nn as nn

from .cfg_cfm import CFGConditionalFlowMatching
from .text_emb import TextEmbedder
from .transformer import TemporalDiT


class HandFlow(nn.Module):
    num_hands = 2
    num_joints = 21
    num_coordinates = 3
    motion_dim = num_hands * num_joints * num_coordinates

    def __init__(
        self,
        hidden_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        text_model: str = "google-t5/t5-small",
        max_text_length: int = 64,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = self.motion_dim

        self.text_embedder = TextEmbedder(
            hidden_dim=hidden_dim,
            model_name=text_model,
            max_length=max_text_length,
        )

        self.transformer = TemporalDiT(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.flow = CFGConditionalFlowMatching(
            model=self.transformer,
            text_embedder=self.text_embedder,
            cfg_dropout=cfg_dropout,
        )

    def forward(
        self,
        joints: torch.Tensor,
        texts: list[str],
        motion_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        motion = self.flatten_joints(joints)
        return self.flow(
            x=motion,
            texts=texts,
            motion_mask=motion_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        texts: list[str],
        num_frames: int = 64,
        num_steps: int = 50,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        motion = self.flow.sample(
            texts=texts,
            num_frames=num_frames,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        return self.unflatten_joints(motion)

    @classmethod
    def flatten_joints(cls, joints: torch.Tensor) -> torch.Tensor:
        if joints.ndim == 3 and joints.shape[-1] == cls.motion_dim:
            return joints

        expected = (cls.num_hands, cls.num_joints, cls.num_coordinates)
        if joints.ndim != 5 or tuple(joints.shape[-3:]) != expected:
            raise ValueError(
                f"joints doit avoir la forme [B, T, 2, 21, 3] ou [B, T, {cls.motion_dim}]"
            )

        return joints.flatten(start_dim=2)

    @classmethod
    def unflatten_joints(cls, motion: torch.Tensor) -> torch.Tensor:
        if motion.ndim != 3 or motion.shape[-1] != cls.motion_dim:
            raise ValueError(f"motion doit avoir la forme [B, T, {cls.motion_dim}]")

        return motion.unflatten(
            dim=-1,
            sizes=(cls.num_hands, cls.num_joints, cls.num_coordinates),
        )

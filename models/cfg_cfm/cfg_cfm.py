
import torch
import torch.nn as nn


class CFGConditionalFlowMatching(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        text_embedder: nn.Module,
        cfg_dropout: float = 0.1,
    ):
        super().__init__()

        self.model = model
        self.text_embedder = text_embedder
        self.cfg_dropout = cfg_dropout

    def forward(self, x, texts, motion_mask=None):
        B = x.shape[0]

        if x.ndim != 3 or x.shape[-1] != self.model.input_dim:
            raise ValueError(
                f"x doit avoir la forme [B, T, {self.model.input_dim}]"
            )
        if len(texts) != B:
            raise ValueError("Le nombre de textes doit correspondre à la taille du batch")
        if motion_mask is not None and motion_mask.shape != x.shape[:2]:
            raise ValueError("motion_mask doit avoir la forme [B, T]")

        t = torch.rand(B, device=x.device)
        noise = torch.randn_like(x)

        if motion_mask is not None:
            valid = motion_mask[..., None].to(x.dtype)
            x = x * valid
            noise = noise * valid

        t_view = t[:, None, None]

        x_t = (1 - t_view) * noise + t_view * x
        target_velocity = x - noise

        drop_mask = (
            torch.rand(B, device=x.device) < self.cfg_dropout
        )

        texts = [
            "" if drop_mask[i].item() else text
            for i, text in enumerate(texts)
        ]

        text_emb, text_mask = self.text_embedder(texts)

        pred_velocity = self.model(
            x_t,
            t,
            text_emb,
            text_mask,
            motion_mask,
        )

        squared_error = (pred_velocity - target_velocity).square()

        if motion_mask is not None:
            valid = motion_mask[..., None].to(squared_error.dtype)
            loss = (squared_error * valid).sum()
            loss = loss / (
                valid.sum() * x.shape[-1]
            ).clamp_min(1)
        else:
            loss = squared_error.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        texts: list[str],
        num_frames: int = 64,
        num_steps: int = 50,
        guidance_scale: float = 2.0,
    ):
        if not texts:
            raise ValueError("texts ne doit pas être vide")
        if num_frames <= 0:
            raise ValueError("num_frames doit être strictement positif")
        if num_steps <= 0:
            raise ValueError("num_steps doit être strictement positif")

        device = next(self.model.parameters()).device
        B = len(texts)

        x = torch.randn(
            B,
            num_frames,
            self.model.input_dim,
            device=device,
        )

        cond_emb, cond_mask = self.text_embedder(texts)

        uncond_emb, uncond_mask = self.text_embedder(
            [""] * B
        )

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full(
                (B,), i * dt, device=device
            )

            v_cond = self.model(
                x, t, cond_emb, cond_mask
            )

            v_uncond = self.model(
                x, t, uncond_emb, uncond_mask
            )

            velocity = v_uncond + guidance_scale * (
                v_cond - v_uncond
            )

            x = x + dt * velocity

        return x

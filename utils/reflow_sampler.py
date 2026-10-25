import torch
from tqdm.auto import tqdm


class ReFlowSampler:
    def __init__(
        self,
        model,
        num_steps,
        guidance_scale=1.0,
        seed=42,
        device=None,
    ):
        if int(num_steps) <= 0:
            raise ValueError("num_steps must be positive")
        self.model = model
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )

    @torch.inference_mode()
    def sample(
        self,
        text_embedding,
        attention_mask,
        frame_mask=None,
        guidance_scale=None,
        seed=None,
        progress=True,
    ):
        single = text_embedding.ndim == 2
        if single:
            text_embedding = text_embedding.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if text_embedding.ndim != 3:
            raise ValueError("text_embedding must have shape [B, L, D]")
        if attention_mask.shape != text_embedding.shape[:2]:
            raise ValueError("attention_mask must have shape [B, L]")

        batch_size = text_embedding.shape[0]
        if frame_mask is None:
            frame_mask = torch.ones(
                batch_size,
                self.model.time_length,
                dtype=torch.bool,
                device=self.device,
            )
        elif frame_mask.ndim == 1:
            frame_mask = frame_mask.unsqueeze(0)
        if frame_mask.shape != (batch_size, self.model.time_length):
            raise ValueError("frame_mask must have shape [B, T]")

        current_seed = self.seed if seed is None else int(seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        text_embedding = text_embedding.to(self.device)
        attention_mask = attention_mask.to(self.device)
        frame_mask = frame_mask.to(self.device)
        x = torch.randn(
            batch_size,
            self.model.in_channels,
            self.model.time_length,
            device=self.device,
        )
        valid = frame_mask[:, None].to(x.dtype)
        x = x * valid
        conditional = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device,
        )
        unconditional = torch.ones_like(conditional)
        scale = (
            self.guidance_scale
            if guidance_scale is None
            else float(guidance_scale)
        )
        dt = 1.0 / self.num_steps

        self.model.eval()
        steps = range(self.num_steps)
        if progress:
            steps = tqdm(steps, desc="ReFlow generation")
        for step in steps:
            timestep = torch.full(
                (batch_size,),
                step * dt,
                device=self.device,
            )
            velocity_conditional = self.model(
                x,
                timestep,
                text_embedding,
                drop_mask=conditional,
                frame_mask=frame_mask,
                attention_mask=attention_mask,
            )
            velocity_unconditional = self.model(
                x,
                timestep,
                text_embedding,
                drop_mask=unconditional,
                frame_mask=frame_mask,
                attention_mask=attention_mask,
            )
            velocity = velocity_unconditional + scale * (
                velocity_conditional - velocity_unconditional
            )
            x = (x + dt * velocity) * valid

        samples = x.transpose(1, 2).cpu()
        return samples[0] if single else samples

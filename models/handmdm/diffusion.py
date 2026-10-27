# Source:
#   DiffuserBase:      https://github.com/leorebensabath/HandMDM/blob/main/src/model/diffusion_base.py
#   GaussianDiffusion: https://github.com/leorebensabath/HandMDM/blob/main/src/model/gaussian.py
#   length_to_mask:    https://github.com/leorebensabath/HandMDM/blob/main/src/data/collate.py
# (HandMDM code is largely inspired by STMC: https://github.com/nv-tlabs/stmc)
#
# Adaptations:
#   - nn.Module instead of LightningModule: training (optimizer, logging, epochs) is done
#     by our own trainer. training_step / validation_step / on_train_epoch_end /
#     configure_optimizers / on_save_checkpoint and the `lr` argument are removed.
#     `device` (a Lightning property) is redefined from the parameters.
#   - STMC / SINC multi-text timeline sampling removed (stmc_forward, p_sample_stmc,
#     sinc_baseline): we only generate from one text per motion. forward = text_forward.
#   - Dead code removed: p_distribution (calls an undefined guided_forward),
#     q_sample_one_x_all_t (debug helper, only einops user), monitor_resources,
#     remove_padding_to_numpy, the debug print in __init__.
#   - The math (schedule constants, q_sample, posteriors, loss, CFG, sampling loop) is
#     unchanged, and so are the attribute names (denoiser, motion_normalizer,
#     text_normalizer): the released checkpoints load with load_state_dict(strict=True).

import torch

from torch import Tensor
from tqdm import tqdm

from .constants import add_constants


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    length = length.to(device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor


def extract(coef, t, tensor):
    shape = tensor.shape
    return coef[t].reshape(shape[0], *((1,) * (len(shape) - 1))).to(tensor)


# Base class for having distribution constants and methods
# Equation taken from this great pdf:
# https://arxiv.org/pdf/2208.11970.pdf
class DiffuserBase(torch.nn.Module):
    def __init__(self, schedule, timesteps):
        super().__init__()

        self.linear_scale = 1.0
        betas = schedule(timesteps)
        # add all the constants (betas / alphas / alpha_cumprod etc)
        add_constants(self, betas, timesteps)

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_from_distribution_function(self, function, *args, noise=None):
        if noise is None:
            noise = torch.randn_like(args[0])

        mean, sigma = function(*args)
        sample = mean + sigma * noise
        return sample

    # Equation 115 (page 15)
    def eps_to_xstart(self, eps, xt, t):
        xstart = (
            extract(self.inv_sqrt_alphas_cumprod, t, xt) * xt
            - extract(self.sqrt_inv_alphas_cumprod_minus_one, t, xt) * eps
        )
        return xstart

    # Equation 133 (page 16)
    def score_to_xstart(self, score, xt, t):
        xstart = (
            extract(self.inv_sqrt_alphas_cumprod, t, xt) * xt
            + extract(self.one_minus_alphas_cumprod_over_sqrt_alphas_cumprod, t, xt)
            * score
        )
        return xstart

    # Equation 151 (page 17)
    def eps_to_score(self, eps, xt, t):
        score = -extract(self.sqrt_inv_one_minus_alphas_cumprod, t, eps) * eps
        return score

    # Equation 69 (page 11) + equation 151 (page 17)
    def xstart_to_score(self, xstart, xt, t):
        score = (
            extract(self.sqrt_alphas_cumprod_over_one_minus_alphas_cumprod, t, xt)
            * xstart
            - extract(self.inv_one_minus_alphas_cumprod, t, xt) * xt
        )
        return score

    # Equation 69 (page 11)
    def xstart_to_eps(self, xstart, xt, t):
        eps = (
            extract(self.sqrt_inv_one_minus_alphas_cumprod, t, xt) * xt
            - extract(
                self.sqrt_alphas_cumprod_over_sqrt_one_minus_alphas_cumprod, t, xt
            )
            * xstart
        )
        return eps

    # Equation 151 (page 17)
    def score_to_eps(self, score, xt, t):
        eps = -extract(self.sqrt_one_minus_alphas_cumprod, t, score) * score
        return eps

    # Equation 70 (page 11)
    def q_distribution(self, xstart, t):
        """
        q(xt | xstart) -> forward diffusion process
        """
        mean = self.linear_scale * extract(self.sqrt_alphas_cumprod, t, xstart) * xstart
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, xstart)
        return mean, sigma

    def q_sample(self, xstart, t, noise=None):
        """
        Sample from q(xt | xstart) -> diffuse the data
        """
        sample = self.sample_from_distribution_function(
            self.q_distribution, xstart, t, noise=noise
        )
        return sample

    # Equation 84 (page 12)
    def q_posterior_distribution_from_xstart_and_xt(self, xstart, xt, t):
        """
        q(xt-1 | xstart, xt) -> Exact denoising distribution (when xstart is known)
        """
        mean = (
            extract(self.posterior_mean_coef1, t, xt) * xstart
            + extract(self.posterior_mean_coef2, t, xt) * xt
        )
        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_xstart_and_xt(self, xstart, xt, t, noise=None):
        """
        Sample from q(xt-1 | xstart, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_xstart_and_xt, xstart, xt, t, noise=noise
        )
        return sample

    # Equation 84 (page 12) + equation 69 (page 11)
    def q_posterior_distribution_from_eps_and_xt(self, eps, xt, t):
        """
        q(xt-1 | eps, xt) -> Exact denoising distribution (when eps is known)
        """
        mean = (
            extract(self.posterior_mean_eps_coef1, t, xt) * xt
            - extract(self.posterior_mean_eps_coef2, t, xt) * eps
        )

        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_eps_and_xt(self, eps, xt, t, noise=None):
        """
        Sample from q(xt-1 | eps, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_eps_and_xt, eps, xt, t, noise=noise
        )
        return sample

    # Equation 84 (page 12) + equation 69 (page 11)
    def q_posterior_distribution_from_score_and_xt(self, score, xt, t):
        """
        q(xt-1 | score, xt) -> Exact denoising distribution (when score is known)
        """
        mean = (
            extract(self.posterior_mean_score_coef1, t, xt) * xt
            + extract(self.posterior_mean_score_coef2, t, xt) * score
        )

        sigma = extract(self.posterior_variance, t, xt)
        return mean, sigma

    def q_posterior_sample_from_score_and_xt(self, score, xt, t, noise=None):
        """
        Sample from q(xt-1 | score, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_score_and_xt, score, xt, t, noise=noise
        )
        return sample

    def q_posterior_distribution_from_output_and_xt(self, output, xt, t):
        if self.prediction == "x":
            return self.q_posterior_distribution_from_xstart_and_xt(output, xt, t)
        elif self.prediction == "eps":
            return self.q_posterior_distribution_from_eps_and_xt(output, xt, t)
        elif self.prediction == "score":
            return self.q_posterior_distribution_from_score_and_xt(output, xt, t)
        else:
            raise NotImplementedError

    def q_posterior_sample_from_output_and_xt(self, output, xt, t, noise=None):
        """
        Sample from q(xt-1 | output, xt)
        """
        sample = self.sample_from_distribution_function(
            self.q_posterior_distribution_from_output_and_xt, output, xt, t, noise=noise
        )
        return sample

    def _format_str(self, string):
        assert string in ["output", "x", "xstart", "eps", "score"]
        if string == "output":
            string = self.prediction
        if string == "x":
            string = "xstart"
        return string

    def to_space(self, space_in, space_out, val, xt, t):
        space_in = self._format_str(space_in)
        space_out = self._format_str(space_out)
        if space_in == space_out:
            return val
        return getattr(self, f"{space_in}_to_{space_out}")(val, xt, t)

    def to_xstart(self, string, val, xt, t):
        return self.to_space(string, "xstart", val, xt, t)

    def to_eps(self, string, val, xt, t):
        return self.to_space(string, "eps", val, xt, t)

    def to_score(self, string, val, xt, t):
        return self.to_space(string, "score", val, xt, t)

    def to_output(self, string, val, xt, t):
        return self.to_space(string, "output", val, xt, t)

    def output_to(self, string, val, xt, t):
        return self.to_space("output", string, val, xt, t)


class GaussianDiffusion(DiffuserBase):
    name = "gaussian"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
    ):
        super().__init__(schedule, timesteps)

        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.prediction = prediction

        self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")

        # normalization
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer

    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"]),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]
        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
            # the condition is already dropped sometimes in the dataloader
        }
        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        t = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample(xstart=x, t=t, noise=noise)
        xt = masked(xt, mask)

        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, t), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t), mask)
        xloss = self.reconstruction_loss(xstart, x)
        loss = {"loss": xloss}
        return loss

    def forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        return self.text_forward(tx_emb, tx_emb_uncond, infos, progress_bar=progress_bar)

    def text_forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
    ):
        # normalize text embeddings first
        device = self.device

        lengths = infos["all_lengths"]
        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, duration, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample(self, xt, y, t):
        # guided forward
        output_cond = self.denoiser(xt, y, t)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xt, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart

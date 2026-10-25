import logging
import time
from itertools import islice

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data.mano import MANO_BIMANUAL_DIM, handx_to_mano_6d


class ReFlowTrainer:
    def __init__(
        self,
        model,
        loaders,
        training_config,
        logger=None,
    ):
        self.model = model
        self.loaders = loaders
        self.config = training_config
        self.logger = logger or self._build_logger()
        self.device = torch.device(
            self._get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.representation = str(self._get("representation", "joints")).lower()
        if self.representation not in {"joints", "mano_6d"}:
            raise ValueError(f"Unknown representation: {self.representation}")
        if self.representation == "mano_6d" and model.in_channels != MANO_BIMANUAL_DIM:
            raise ValueError("The MANO 6D model must have 198 channels")
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()
        self.history = {"train": [], "test": None}
        self.global_step = 0

    def _get(self, name, default=None):
        if isinstance(self.config, dict):
            return self.config.get(name, default)
        return getattr(self.config, name, default)

    def _build_logger(self):
        logger = logging.getLogger("ReFlowTrainer")
        level = getattr(
            logging,
            self._get("log_level", "INFO").upper(),
            logging.INFO,
        )
        logger.setLevel(level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(handler)

        return logger

    def _channels_first(self, motion):
        if motion.ndim == 5:
            return motion.flatten(start_dim=2).transpose(1, 2)
        if motion.ndim == 3 and motion.shape[-1] == self.model.in_channels:
            return motion.transpose(1, 2)
        if motion.ndim == 3 and motion.shape[1] == self.model.in_channels:
            return motion
        raise ValueError("Invalid motion shape")

    def _batch_to_target(self, batch):
        if self.representation == "joints":
            if "motion" not in batch:
                raise KeyError("Batch does not contain motion")
            return batch["motion"]

        if "x" in batch:
            return batch["x"]

        if "mano_motion" in batch:
            return batch["mano_motion"]

        required = ("left_pose", "left_trans", "right_pose", "right_trans")
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"Missing MANO parameters: {missing}")
        return handx_to_mano_6d(
            batch["left_pose"],
            batch["left_trans"],
            batch["right_pose"],
            batch["right_trans"],
        )

    def _build_optimizer(self):
        name = self._get("optimizer", "adamw").lower()
        lr = self._get("lr", self._get("learning_rate", 1e-4))
        weight_decay = self._get("weight_decay", 0.0)
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=self._get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        name = self._get("scheduler", None)
        if name is None or str(name).lower() == "none":
            return None

        name = str(name).lower()
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self._get("step_size", 10),
                gamma=self._get("gamma", 0.5),
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self._get("epochs", 1),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def _build_criterion(self):
        name = self._get("criterion", "mse").lower()
        if name == "mse":
            return nn.MSELoss(reduction="none")
        if name == "l1":
            return nn.L1Loss(reduction="none")
        raise ValueError(f"Unknown criterion: {name}")

    def _prepare_batch(self, batch):
        motion = self._channels_first(self._batch_to_target(batch))
        if motion.shape[-1] != self.model.time_length:
            raise ValueError(
                f"Temporal length {motion.shape[-1]} differs from "
                f"model.time_length={self.model.time_length}"
            )
        motion = motion.to(self.device, non_blocking=True)

        if "text_embedding" not in batch:
            raise KeyError("Batch does not contain text_embedding")
        text_embedding = batch["text_embedding"].to(
            self.device,
            non_blocking=True,
        )
        if "attention_mask" not in batch:
            raise KeyError("Batch does not contain attention_mask")
        attention_mask = batch["attention_mask"].to(
            self.device,
            non_blocking=True,
        )
        frame_mask = batch.get("frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(self.device, non_blocking=True)

        return motion.contiguous(), text_embedding, attention_mask, frame_mask

    def _compute_loss(self, batch):
        x_1, text_embedding, attention_mask, frame_mask = self._prepare_batch(
            batch
        )
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        timesteps = torch.rand(batch_size, device=self.device)
        t = timesteps[:, None, None]

        if frame_mask is not None:
            valid = frame_mask[:, None, :].to(x_1.dtype)
            x_1 = x_1 * valid
            x_0 = x_0 * valid

        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        drop_mask = (
            torch.rand(batch_size, device=self.device)
            < self._get("cfg_dropout", 0.1)
        )
        prediction = self.model(
            x_t,
            timesteps,
            text_embedding,
            drop_mask=drop_mask,
            frame_mask=frame_mask,
            attention_mask=attention_mask,
        )
        flow_loss = self.criterion(prediction, target_velocity)

        if frame_mask is None:
            flow_loss = flow_loss.mean()
        else:
            flow_loss = flow_loss * valid
            flow_loss = flow_loss.sum() / (
                valid.sum() * x_1.shape[1]
            ).clamp_min(1)

        return flow_loss

    def _run_epoch(self, epoch=1, epochs=1):
        loader = self.loaders["train"]
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        log_every = max(1, self._get("log_every", 10))
        max_steps = self._get("max_steps", None)
        total_batches = len(loader)
        if max_steps is not None:
            total_batches = min(total_batches, max_steps)

        progress = tqdm(
            islice(loader, total_batches),
            total=total_batches,
            desc=f"Train {epoch}/{epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._compute_loss(batch)
            loss.backward()

            grad_clip = self._get("grad_clip", None)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()
            self.global_step += 1
            batch_size = batch["text_embedding"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            average_loss = total_loss / max(total_samples, 1)
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                average=f"{average_loss:.6f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

            if batch_index % log_every == 0 or batch_index == total_batches:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "epoch=%d/%d batch=%d/%d global_step=%d loss=%.6f lr=%.2e",
                    epoch,
                    epochs,
                    batch_index,
                    total_batches,
                    self.global_step,
                    average_loss,
                    lr,
                )

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _evaluate(self, loader, split):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        log_every = max(1, self._get("log_every", 10))
        total_batches = len(loader)

        progress = tqdm(
            loader,
            total=total_batches,
            desc=split.capitalize(),
            unit="batch",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress, start=1):
            loss = self._compute_loss(batch)
            batch_size = batch["text_embedding"].shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            average_loss = total_loss / max(total_samples, 1)
            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                average=f"{average_loss:.6f}",
            )

            if batch_index % log_every == 0 or batch_index == total_batches:
                self.logger.info(
                    "%s batch=%d/%d loss=%.6f",
                    split,
                    batch_index,
                    total_batches,
                    average_loss,
                )

        return total_loss / max(total_samples, 1)

    def _test_epoch(self):
        loader = self.loaders.get("test")
        if loader is None:
            return None
        return self._evaluate(loader, "test")

    def run(self):
        epochs = self._get("epochs", 1)
        max_steps = self._get("max_steps", None)
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.logger.info(
            "start device=%s representation=%s epochs=%d "
            "max_steps_per_epoch=%s train_batches=%d parameters=%d",
            self.device,
            self.representation,
            epochs,
            max_steps,
            len(self.loaders["train"]),
            trainable_parameters,
        )

        for epoch in range(epochs):
            start_time = time.perf_counter()
            train_loss = self._run_epoch(epoch + 1, epochs)
            self.history["train"].append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.perf_counter() - start_time
            message = (
                f"epoch={epoch + 1}/{epochs} train={train_loss:.6f} "
                f"duration={elapsed:.1f}s"
            )
            self.logger.info(message)

        self.history["test"] = self._test_epoch()
        if self.history["test"] is not None:
            self.logger.info("test=%.6f", self.history["test"])

        self.logger.info("training completed")

        return self.history

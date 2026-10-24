import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data import get_handx_loaders
from models import TemporalUnet1D
from utils.reflow_parser import parse_reflow_config
from utils.reflow_trainer import ReFlowTrainer


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "training" / "reflow_config.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_checkpoint(path, config, model, trainer, history, dataset):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict()
                if trainer.scheduler is not None
                else None
            ),
            "global_step": trainer.global_step,
            "history": history,
            "config": config,
            "motion_mean": dataset.motion_mean,
            "motion_std": dataset.motion_std,
            "text_mean": dataset.text_mean,
            "text_std": dataset.text_std,
            "metadata": dataset.metadata,
        },
        path,
    )
    return path


def main():
    config = parse_reflow_config(load_config())
    print(yaml.safe_dump(config, sort_keys=False))
    set_seed(int(config.get("seed", 42)))

    data_config = dict(config["data"])
    data_config["target_length"] = int(config["model"]["time_length"])
    if data_config.get("root") is not None:
        data_config["root"] = resolve_path(data_config["root"])

    loaders = get_handx_loaders(**data_config)
    model = TemporalUnet1D(**config["model"])
    trainer = ReFlowTrainer(
        model=model,
        loaders=loaders,
        training_config=config["training"],
    )
    history = trainer.run()

    checkpoint_path = save_checkpoint(
        config["training"]["checkpoint_path"],
        config,
        model,
        trainer,
        history,
        loaders["train"].dataset,
    )
    trainer.logger.info("checkpoint saved to %s", checkpoint_path)
    return history


if __name__ == "__main__":
    main()

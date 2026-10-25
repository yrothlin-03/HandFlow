import os
import pickle
from functools import partial
from pathlib import Path

import lmdb
import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "handx.yaml"
VALID_SPLITS = {"train", "test"}
_ENVIRONMENTS = {}
TEMPORAL_FIELDS = {
    "x",
    "frame_mask",
    "left_shape",
    "right_shape",
    "joint_positions",
}


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def open_environment(path, readahead=False):
    key = (os.getpid(), str(path))
    if key not in _ENVIRONMENTS:
        _ENVIRONMENTS[key] = lmdb.open(
            str(path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=readahead,
            meminit=False,
            max_readers=512,
        )
    return _ENVIRONMENTS[key]


def read_database_header(path, split, readahead=False):
    environment = lmdb.open(
        str(path),
        subdir=True,
        readonly=True,
        lock=False,
        readahead=readahead,
        meminit=False,
        max_readers=512,
    )
    try:
        with environment.begin() as transaction:
            metadata_value = transaction.get(b"__metadata__")
            statistics_value = transaction.get(b"__statistics__")
            keys_value = transaction.get(f"__keys__/{split}".encode("utf-8"))
            if metadata_value is None:
                raise KeyError("Missing __metadata__ key in LMDB")
            if statistics_value is None:
                raise KeyError("Missing __statistics__ key in LMDB")
            if keys_value is None:
                raise KeyError(f"Missing __keys__/{split} key in LMDB")
            return (
                pickle.loads(metadata_value),
                pickle.loads(statistics_value),
                pickle.loads(keys_value),
            )
    finally:
        environment.close()


class HandX(Dataset):
    def __init__(
        self,
        split,
        root=None,
        use_paraphrases=True,
        normalize_motion=False,
        normalize_text=False,
    ):
        split = str(split).lower()
        if split not in VALID_SPLITS:
            raise ValueError("split must be 'train' or 'test'")

        config = load_config()
        self.root = (
            Path(root).resolve()
            if root is not None
            else resolve_path(config["paths"]["output"])
        )
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        self.split = split
        self.use_paraphrases = bool(use_paraphrases)
        self.normalize_motion = bool(normalize_motion)
        self.normalize_text = bool(normalize_text)
        self.readahead = bool(config["lmdb"].get("readahead", False))
        self.metadata, self.statistics, self.keys = read_database_header(
            self.root,
            self.split,
            self.readahead,
        )

        self.motion_mean = torch.as_tensor(
            self.statistics["mean"], dtype=torch.float32
        )
        self.motion_std = torch.as_tensor(
            self.statistics["std"], dtype=torch.float32
        )
        self.text_mean = torch.as_tensor(
            self.statistics["text_mean"], dtype=torch.float32
        )
        self.text_std = torch.as_tensor(
            self.statistics["text_std"], dtype=torch.float32
        )

        if self.motion_mean.shape != (198,) or self.motion_std.shape != (198,):
            raise ValueError("Invalid MANO statistics")
        text_dimension = int(self.metadata["text"]["dimension"])
        if (
            self.text_mean.shape != (text_dimension,)
            or self.text_std.shape != (text_dimension,)
        ):
            raise ValueError("Invalid text statistics")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        environment = open_environment(self.root, self.readahead)
        with environment.begin(buffers=False) as transaction:
            value = transaction.get(key.encode("utf-8"))
        if value is None:
            raise KeyError(f"Missing LMDB entry: {key}")
        record = pickle.loads(value)

        text_count = len(record["texts"])
        if self.split == "train" and self.use_paraphrases:
            text_index = int(torch.randint(text_count, ()).item())
        else:
            text_index = 0

        x = self._tensor(record["x"])
        text_embeddings = self._tensor(record["text_embeddings"])
        attention_masks = torch.as_tensor(
            record["attention_masks"], dtype=torch.bool
        )
        if self.normalize_motion:
            x = (x - self.motion_mean) / self.motion_std
        if self.normalize_text:
            text_embeddings = (text_embeddings - self.text_mean) / self.text_std

        positions = self._tensor(record["joint_positions"])
        frame_mask = torch.as_tensor(record["frame_mask"], dtype=torch.bool)
        length = int(record["length"])
        if x.shape != (length, 198):
            raise ValueError(f"Invalid MANO features for {key}: {tuple(x.shape)}")
        if positions.shape != (length, 2, 21, 3):
            raise ValueError(
                f"Invalid joint positions for {key}: {tuple(positions.shape)}"
            )

        output = {
            "id": record["id"],
            "dataset": record["dataset"],
            "split": record["split"],
            "index": torch.tensor(record["index"], dtype=torch.long),
            "source_key": record["source_key"],
            "source": record["source"],
            "x": x,
            "length": torch.tensor(length, dtype=torch.long),
            "frame_mask": frame_mask,
            "is_left_hand": torch.tensor(record["is_left_hand"]),
            "is_right_hand": torch.tensor(record["is_right_hand"]),
            "left_shape": self._tensor(record["left_shape"]),
            "right_shape": self._tensor(record["right_shape"]),
            "joint_positions": positions,
            "texts": record["texts"],
            "text_embeddings": text_embeddings,
            "attention_masks": attention_masks,
            "text_index": torch.tensor(text_index, dtype=torch.long),
            "text": record["texts"][text_index],
            "text_embedding": text_embeddings[text_index],
            "attention_mask": attention_masks[text_index],
            "canonical_text": record["canonical_text"],
            "left_text": record["left_texts"][text_index],
            "right_text": record["right_texts"][text_index],
            "interaction_text": record["interaction_texts"][text_index],
            "motion_normalized": torch.tensor(self.normalize_motion),
            "text_normalized": torch.tensor(self.normalize_text),
        }
        return output

    @staticmethod
    def _tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32)


def _pad_time(tensor, target_length):
    current_length = tensor.shape[1]
    if current_length > target_length:
        raise ValueError(
            f"Sequence length {current_length} exceeds target_length={target_length}"
        )
    if current_length == target_length:
        return tensor
    padding = tensor.new_zeros(
        tensor.shape[0],
        target_length - current_length,
        *tensor.shape[2:],
    )
    return torch.cat((tensor, padding), dim=1)


def collate_handx(records, target_length=None):
    if not records:
        raise ValueError("HandX batch must not be empty")
    if target_length is not None:
        if isinstance(target_length, bool) or int(target_length) <= 0:
            raise ValueError("target_length must be a positive integer")
        target_length = int(target_length)

    batch = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if key in TEMPORAL_FIELDS:
            batch[key] = pad_sequence(values, batch_first=True)
        elif key in {"text_embeddings", "attention_masks"}:
            batch[key] = pad_sequence(values, batch_first=True)
        elif torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values

    if target_length is not None:
        for key in TEMPORAL_FIELDS:
            batch[key] = _pad_time(batch[key], target_length)

    text_counts = torch.tensor(
        [len(record["texts"]) for record in records],
        dtype=torch.long,
    )
    batch["text_count"] = text_counts
    batch["text_mask"] = (
        torch.arange(int(text_counts.max())).unsqueeze(0)
        < text_counts.unsqueeze(1)
    )
    return batch


def _get_handx_loader(
    split,
    batch_size=32,
    root=None,
    use_paraphrases=True,
    normalize_motion=False,
    normalize_text=False,
    target_length=None,
    shuffle=None,
    num_workers=0,
    drop_last=False,
    pin_memory=None,
):
    dataset = HandX(
        split=split,
        root=root,
        use_paraphrases=use_paraphrases,
        normalize_motion=normalize_motion,
        normalize_text=normalize_text,
    )
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=partial(collate_handx, target_length=target_length),
    )


def get_handx_loaders(
    batch_size=32,
    root=None,
    use_paraphrases=True,
    normalize_motion=False,
    normalize_text=False,
    target_length=None,
    num_workers=0,
    drop_last_train=False,
    pin_memory=None,
):
    train_loader = _get_handx_loader(
        split="train",
        batch_size=batch_size,
        root=root,
        use_paraphrases=use_paraphrases,
        normalize_motion=normalize_motion,
        normalize_text=normalize_text,
        target_length=target_length,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last_train,
        pin_memory=pin_memory,
    )
    test_loader = _get_handx_loader(
        split="test",
        batch_size=batch_size,
        root=root,
        use_paraphrases=False,
        normalize_motion=normalize_motion,
        normalize_text=normalize_text,
        target_length=target_length,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return {"train": train_loader, "test": test_loader}

import io
import json
import pickle
import shutil
import sys
import time
import zipfile
from pathlib import Path

import lmdb
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, T5EncoderModel
from tqdm.auto import tqdm

try:
    from .mano import MANO_BIMANUAL_DIM, handx_to_mano_6d
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.mano import MANO_BIMANUAL_DIM, handx_to_mano_6d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "handx.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_npz_item(archive, key):
    return np.load(
        io.BytesIO(archive.read(f"{key}.npy")),
        allow_pickle=True,
    ).item()


def archive_keys(archive):
    return sorted(
        (
            Path(name).stem
            for name in archive.namelist()
            if name.endswith(".npy")
        ),
        key=int,
    )


def load_source_metadata(root):
    path = root / "handx_to_dataset_source_release.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def source_for(metadata, split, key):
    entries = metadata.get(split, {})
    if isinstance(entries, dict):
        return entries.get(key, "unknown")
    index = int(key)
    return entries[index] if index < len(entries) else "unknown"


def combined_texts(motion):
    left = list(motion["left_annotation"])
    right = list(motion["right_annotation"])
    interaction = list(motion["interaction_annotation"])
    count = min(len(left), len(right), len(interaction))
    if count == 0:
        raise ValueError("Missing text annotations")
    left = left[:count]
    right = right[:count]
    interaction = interaction[:count]
    texts = [
        f"Left hand: {left[index]} Right hand: {right[index]} "
        f"Interaction: {interaction[index]}"
        for index in range(count)
    ]
    return left, right, interaction, texts


class TextEncoder:
    def __init__(self, model_name, device, max_length, dtype):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.dimension = int(self.model.config.d_model)

    def __call__(self, texts):
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            ).last_hidden_state
        return (
            hidden.cpu().numpy().astype(self.dtype, copy=False),
            tokens["attention_mask"].cpu().numpy().astype(np.bool_, copy=False),
        )


def prepare_record(split, index, key, motion, mano, source):
    positions = np.asarray(motion["motion"], dtype=np.float32)
    left_pose = np.asarray(mano["left_pose"], dtype=np.float32)
    right_pose = np.asarray(mano["right_pose"], dtype=np.float32)
    left_trans = np.asarray(mano["left_trans"], dtype=np.float32)
    right_trans = np.asarray(mano["right_trans"], dtype=np.float32)
    left_shape = np.asarray(mano["left_shape"], dtype=np.float32)
    right_shape = np.asarray(mano["right_shape"], dtype=np.float32)
    length = int(left_pose.shape[0])

    if left_pose.shape != (length, 48) or right_pose.shape != (length, 48):
        raise ValueError(f"Invalid MANO poses for {split}/{key}")
    if left_trans.shape != (length, 3) or right_trans.shape != (length, 3):
        raise ValueError(f"Invalid MANO translations for {split}/{key}")
    if left_shape.shape != (length, 10) or right_shape.shape != (length, 10):
        raise ValueError(f"Invalid MANO shapes for {split}/{key}")
    if positions.shape != (length, 2, 21, 3):
        raise ValueError(f"Invalid joint positions for {split}/{key}")

    x = handx_to_mano_6d(
        torch.from_numpy(left_pose),
        torch.from_numpy(left_trans),
        torch.from_numpy(right_pose),
        torch.from_numpy(right_trans),
    ).numpy()
    left_texts, right_texts, interaction_texts, texts = combined_texts(motion)
    identifier = f"{split}/{index:08d}"

    return {
        "id": identifier,
        "dataset": "handx",
        "split": split,
        "index": index,
        "source_key": key,
        "source": source,
        "x": x.astype(np.float32, copy=False),
        "length": length,
        "frame_mask": np.ones(length, dtype=np.bool_),
        "is_left_hand": True,
        "is_right_hand": True,
        "left_shape": left_shape,
        "right_shape": right_shape,
        "joint_positions": positions,
        "left_texts": left_texts,
        "right_texts": right_texts,
        "interaction_texts": interaction_texts,
        "texts": texts,
        "text": texts[0],
        "canonical_text": texts[0],
    }


def encode_records(records, encoder, batch_size):
    texts = []
    offsets = [0]
    for record in records:
        texts.extend(record["texts"])
        offsets.append(len(texts))

    embedding_parts = []
    attention_mask_parts = []
    for start in range(0, len(texts), batch_size):
        embeddings, attention_masks = encoder(texts[start : start + batch_size])
        embedding_parts.append(embeddings)
        attention_mask_parts.append(attention_masks)
    embeddings = np.concatenate(embedding_parts, axis=0)
    attention_masks = np.concatenate(attention_mask_parts, axis=0)

    for record, start, end in zip(records, offsets[:-1], offsets[1:]):
        record["text_embeddings"] = embeddings[start:end].copy()
        record["attention_masks"] = attention_masks[start:end].copy()
        record["text_embedding"] = embeddings[start].copy()
        record["attention_mask"] = attention_masks[start].copy()


def feature_schema(text_encoder, embedding_dtype, statistics_config):
    return {
        "version": 1,
        "serialization": "pickle",
        "dataset": "HandX",
        "feature_dimension": MANO_BIMANUAL_DIM,
        "feature_dtype": "float32",
        "feature_shape": ["T", MANO_BIMANUAL_DIM],
        "joint_positions_shape": ["T", 2, 21, 3],
        "feature_layout": {
            "left": {
                "range": [0, 99],
                "translation": [0, 3],
                "rotation_6d": [3, 99],
                "rotation_shape": [16, 6],
            },
            "right": {
                "range": [99, 198],
                "translation": [99, 102],
                "rotation_6d": [102, 198],
                "rotation_shape": [16, 6],
            },
        },
        "rotation_6d": {
            "convention": "pytorch3d_handmdm",
            "matrix_values": ["r00", "r01", "r02", "r10", "r11", "r12"],
            "joint_count_per_hand": 16,
            "root_joint_index": 0,
        },
        "text": {
            "model": text_encoder.model_name,
            "dimension": text_encoder.dimension,
            "dtype": embedding_dtype,
            "max_length": text_encoder.max_length,
            "pooling": "none",
        },
        "statistics": {
            "split": statistics_config["split"],
            "motion_fields": ["count", "mean", "std"],
            "text_fields": ["text_count", "text_mean", "text_std"],
            "text_scope": statistics_config["text_scope"],
        },
        "record_fields": [
            "id",
            "dataset",
            "split",
            "index",
            "source_key",
            "source",
            "x",
            "length",
            "frame_mask",
            "is_left_hand",
            "is_right_hand",
            "left_shape",
            "right_shape",
            "joint_positions",
            "left_texts",
            "right_texts",
            "interaction_texts",
            "texts",
            "text",
            "canonical_text",
            "text_embeddings",
            "attention_masks",
            "text_embedding",
            "attention_mask",
        ],
    }


def write_records(transaction, records, keys, split_keys, statistics):
    for record in records:
        key = record["id"]
        transaction.put(
            key.encode("utf-8"),
            pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
            overwrite=False,
        )
        keys.append(key)
        split_keys[record["split"]].append(key)
        if record["split"] == statistics["split"]:
            values = record["x"].astype(np.float64)
            statistics["sum"] += values.sum(axis=0)
            statistics["sum_square"] += np.square(values).sum(axis=0)
            statistics["count"] += len(values)
            text_values = record["text_embeddings"][
                record["attention_masks"]
            ].astype(np.float64)
            statistics["text_sum"] += text_values.sum(axis=0)
            statistics["text_sum_square"] += np.square(text_values).sum(axis=0)
            statistics["text_count"] += len(text_values)


def process_split(
    root,
    split,
    source_metadata,
    encoder,
    text_batch_size,
    transaction,
    environment,
    commit_every,
    keys,
    split_keys,
    statistics,
):
    motion_path = root / f"{split}_can_pos_all_wotextfeat.npz"
    mano_path = root / f"{split}_mano.npz"
    with zipfile.ZipFile(motion_path) as motion_archive, zipfile.ZipFile(
        mano_path
    ) as mano_archive:
        motion_keys = archive_keys(motion_archive)
        mano_keys = set(archive_keys(mano_archive))
        if any(key not in mano_keys for key in motion_keys):
            raise ValueError(f"Missing MANO keys in {split} split")

        pending = []
        written_since_commit = 0
        progress = tqdm(
            enumerate(motion_keys),
            total=len(motion_keys),
            desc=f"HandX {split}",
            unit="sequence",
            dynamic_ncols=True,
        )
        for index, key in progress:
            pending.append(
                prepare_record(
                    split,
                    index,
                    key,
                    load_npz_item(motion_archive, key),
                    load_npz_item(mano_archive, key),
                    source_for(source_metadata, split, key),
                )
            )
            if len(pending) < text_batch_size and index + 1 < len(motion_keys):
                continue

            encode_records(pending, encoder, text_batch_size)
            write_records(transaction, pending, keys, split_keys, statistics)
            written_since_commit += len(pending)
            pending.clear()

            if written_since_commit >= commit_every:
                transaction.commit()
                transaction = environment.begin(write=True)
                written_since_commit = 0

    return transaction


def main():
    config = load_config()
    text_config = config["text_embedding"]
    lmdb_config = config["lmdb"]
    feature_config = config["features"]
    statistics_config = config["statistics"]
    splits = list(config["splits"])
    root = resolve_path(config["paths"]["input"])
    output = resolve_path(config["paths"]["output"])

    if not root.is_dir():
        raise FileNotFoundError(root)
    if int(text_config["batch_size"]) <= 0 or int(lmdb_config["commit_every"]) <= 0:
        raise ValueError("Batch and commit sizes must be positive")
    if text_config["dtype"] not in {"float16", "float32"}:
        raise ValueError("Embedding dtype must be float16 or float32")
    if text_config["pooling"] != "none":
        raise ValueError("Text pooling must be none")
    if int(feature_config["dimension"]) != MANO_BIMANUAL_DIM:
        raise ValueError("Configured MANO dimension must be 198")
    if statistics_config["split"] not in splits:
        raise ValueError("Statistics split is missing from configured splits")
    if output.exists():
        if not lmdb_config["overwrite"]:
            raise FileExistsError(
                f"{output} already exists; enable lmdb.overwrite in {CONFIG_PATH}"
            )
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = text_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = TextEncoder(
        text_config["model_name"],
        device,
        int(text_config["max_length"]),
        text_config["dtype"],
    )
    source_metadata = load_source_metadata(root)
    environment = lmdb.open(
        str(output),
        subdir=True,
        map_size=int(lmdb_config["map_size_gb"]) * 1024**3,
        readonly=False,
        lock=True,
        readahead=bool(lmdb_config["readahead"]),
        meminit=bool(lmdb_config["meminit"]),
        map_async=bool(lmdb_config["map_async"]),
    )
    transaction = environment.begin(write=True)
    keys = []
    split_keys = {split: [] for split in splits}
    statistics = {
        "split": statistics_config["split"],
        "sum": np.zeros(MANO_BIMANUAL_DIM, dtype=np.float64),
        "sum_square": np.zeros(MANO_BIMANUAL_DIM, dtype=np.float64),
        "count": 0,
        "text_sum": np.zeros(encoder.dimension, dtype=np.float64),
        "text_sum_square": np.zeros(encoder.dimension, dtype=np.float64),
        "text_count": 0,
    }
    started_at = time.time()

    try:
        for split in splits:
            transaction = process_split(
                root,
                split,
                source_metadata,
                encoder,
                int(text_config["batch_size"]),
                transaction,
                environment,
                int(lmdb_config["commit_every"]),
                keys,
                split_keys,
                statistics,
            )

        count = statistics["count"]
        if count == 0 or statistics["text_count"] == 0:
            raise ValueError("No data available to compute statistics")
        epsilon = float(statistics_config["epsilon"])
        mean = statistics["sum"] / count
        variance = statistics["sum_square"] / count - np.square(mean)
        std = np.sqrt(np.maximum(variance, epsilon))
        text_count = statistics["text_count"]
        text_mean = statistics["text_sum"] / text_count
        text_variance = (
            statistics["text_sum_square"] / text_count - np.square(text_mean)
        )
        text_std = np.sqrt(np.maximum(text_variance, epsilon))
        stats = {
            "count": count,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "text_count": text_count,
            "text_mean": text_mean.astype(np.float32),
            "text_std": text_std.astype(np.float32),
        }
        metadata = feature_schema(
            encoder,
            text_config["dtype"],
            statistics_config,
        )
        metadata.update(
            {
                "config_path": str(CONFIG_PATH),
                "config": config,
                "length": len(keys),
                "splits": {name: len(values) for name, values in split_keys.items()},
                "keys_key": "__keys__",
                "split_keys": {
                    split: f"__keys__/{split}" for split in split_keys
                },
                "statistics_key": "__statistics__",
                "elapsed_seconds": time.time() - started_at,
            }
        )
        transaction.put(b"__metadata__", pickle.dumps(metadata))
        transaction.put(b"__keys__", pickle.dumps(keys))
        for split, values in split_keys.items():
            transaction.put(
                f"__keys__/{split}".encode("utf-8"),
                pickle.dumps(values),
            )
        transaction.put(b"__statistics__", pickle.dumps(stats))
        transaction.commit()
        environment.sync()
    except BaseException:
        try:
            transaction.abort()
        except lmdb.Error:
            pass
        raise
    finally:
        environment.close()

    print(f"LMDB: {output}")
    print(f"Entries: {len(keys)}")
    for split, values in split_keys.items():
        print(f"{split}: {len(values)}")
    print(f"Features: {MANO_BIMANUAL_DIM}")
    print(f"Text embedding: {encoder.dimension}")


if __name__ == "__main__":
    main()

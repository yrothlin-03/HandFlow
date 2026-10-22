import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HANDX_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "HandX"


class HandX(Dataset):
    def __init__(
        self,
        root=None,
        train=True,
        include_mano=False,
        use_paraphrases=True,
        text_mode="combined",
    ):
        self.root = Path(root) if root is not None else HANDX_ROOT
        self.train = bool(train)
        self.split = "train" if self.train else "test"
        self.include_mano = bool(include_mano)
        self.use_paraphrases = bool(use_paraphrases)
        self.text_mode = text_mode.lower()

        valid_text_modes = {"combined", "interaction", "left", "right"}
        if self.text_mode not in valid_text_modes:
            raise ValueError(f"text_mode doit être parmi {tuple(sorted(valid_text_modes))}")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        self.motion_path = self.root / f"{self.split}_can_pos_all_wotextfeat.npz"
        self.mano_path = self.root / f"{self.split}_mano.npz"
        if not self.motion_path.is_file():
            raise FileNotFoundError(self.motion_path)
        if self.include_mano and not self.mano_path.is_file():
            raise FileNotFoundError(self.mano_path)

        with zipfile.ZipFile(self.motion_path) as archive:
            self.keys = sorted(
                (Path(name).stem for name in archive.namelist()),
                key=int,
            )

        self.sources = self._load_sources()
        self._motion_archive = None
        self._mano_archive = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        sample = self._load_sample("motion", key)
        motion = torch.as_tensor(
            np.asarray(sample["motion"], dtype=np.float32).copy()
        )

        if motion.shape != (60, 2, 21, 3):
            raise ValueError(f"Mouvement HandX invalide pour la clé {key}: {motion.shape}")

        annotation_index = self._annotation_index(sample)
        left_text = sample["left_annotation"][annotation_index]
        right_text = sample["right_annotation"][annotation_index]
        interaction_text = sample["interaction_annotation"][annotation_index]
        text = self._format_text(left_text, right_text, interaction_text)

        canonical_left = sample["left_annotation"][0]
        canonical_right = sample["right_annotation"][0]
        canonical_interaction = sample["interaction_annotation"][0]
        canonical_text = self._format_text(
            canonical_left,
            canonical_right,
            canonical_interaction,
        )

        output = {
            "motion": motion,
            "j_lhand": motion[:, 0],
            "j_rhand": motion[:, 1],
            "frame_mask": torch.ones(60, dtype=torch.bool),
            "lhand_mask": torch.ones(60, dtype=torch.bool),
            "rhand_mask": torch.ones(60, dtype=torch.bool),
            "is_lhand": torch.tensor(True),
            "is_rhand": torch.tensor(True),
            "nframes": torch.tensor(60, dtype=torch.long),
            "text": text,
            "canonical_text": canonical_text,
            "left_text": left_text,
            "right_text": right_text,
            "interaction_text": interaction_text,
            "source": self.sources[index],
            "dataset_name": "handx",
            "key": key,
            "index": torch.tensor(index, dtype=torch.long),
        }

        if self.include_mano:
            mano = self._load_sample("mano", key)
            output.update(
                {
                    "left_shape": self._tensor(mano["left_shape"]),
                    "right_shape": self._tensor(mano["right_shape"]),
                    "left_pose": self._tensor(mano["left_pose"]),
                    "right_pose": self._tensor(mano["right_pose"]),
                    "left_trans": self._tensor(mano["left_trans"]),
                    "right_trans": self._tensor(mano["right_trans"]),
                }
            )

        return output

    def _annotation_index(self, sample):
        lengths = [
            len(sample["left_annotation"]),
            len(sample["right_annotation"]),
            len(sample["interaction_annotation"]),
        ]
        count = min(lengths)
        if count == 0:
            raise ValueError("Une séquence HandX ne contient aucune annotation")
        if self.train and self.use_paraphrases:
            return int(np.random.randint(count))
        return 0

    def _format_text(self, left, right, interaction):
        if self.text_mode == "left":
            return left
        if self.text_mode == "right":
            return right
        if self.text_mode == "interaction":
            return interaction
        return f"Left hand: {left} Right hand: {right} Interaction: {interaction}"

    def _load_sample(self, kind, key):
        archive = self._get_archive(kind)
        member_name = f"{key}.npy"
        try:
            data = archive.read(member_name)
        except KeyError as error:
            raise KeyError(f"Clé {key} absente de l'archive {kind}") from error
        return np.load(io.BytesIO(data), allow_pickle=True).item()

    def _get_archive(self, kind):
        attribute = f"_{kind}_archive"
        archive = getattr(self, attribute)
        if archive is None:
            path = self.motion_path if kind == "motion" else self.mano_path
            archive = zipfile.ZipFile(path)
            setattr(self, attribute, archive)
        return archive

    def _load_sources(self):
        metadata_path = self.root / "handx_to_dataset_source_release.json"
        if not metadata_path.is_file():
            return ["unknown"] * len(self.keys)

        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)

        entries = metadata.get(self.split, {})
        if isinstance(entries, dict):
            sources = [entries[key] for key in sorted(entries, key=int)]
        else:
            sources = list(entries)

        sources = [source for source in sources if source not in {"ARCTIC", "H2O"}]
        if len(sources) != len(self.keys):
            return ["unknown"] * len(self.keys)
        return sources

    def close(self):
        for name in ("_motion_archive", "_mano_archive"):
            archive = getattr(self, name, None)
            if archive is not None:
                archive.close()
                setattr(self, name, None)

    def __getstate__(self):
        self.close()
        return self.__dict__.copy()

    def __del__(self):
        self.close()

    @staticmethod
    def _tensor(values):
        return torch.as_tensor(np.asarray(values, dtype=np.float32).copy())


def get_handx_loader(
    batch_size=32,
    shuffle=True,
    root=None,
    train=True,
    include_mano=False,
    use_paraphrases=True,
    text_mode="combined",
    num_workers=0,
    drop_last=False,
):
    dataset = HandX(
        root=root,
        train=train,
        include_mano=include_mano,
        use_paraphrases=use_paraphrases,
        text_mode=text_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

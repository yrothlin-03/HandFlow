import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TEXT2HOI_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "Text2HOI"
PATHS = {name: TEXT2HOI_ROOT / name for name in ("arctic", "grab", "h2o")}


def _npz_array(path, key):
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED:
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data[key]).copy()
        with archive.open(info) as member:
            version = np.lib.format.read_magic(member)
            shape, fortran_order, dtype = np.lib.format._read_array_header(member, version)
            npy_header_size = member.tell()

    with open(path, "rb") as file:
        file.seek(info.header_offset)
        header = struct.unpack("<IHHHHHIIIHH", file.read(30))

    if header[0] != 0x04034B50:
        raise ValueError(f"Archive NPZ invalide: {path}")

    offset = info.header_offset + 30 + header[-2] + header[-1] + npy_header_size
    order = "F" if fortran_order else "C"
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset, order=order)


class Text2HOI(Dataset):
    def __init__(
        self,
        dataset_name="arctic",
        root=None,
        max_frames=150,
        train=True,
        use_paraphrases=True,
    ):
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in PATHS:
            raise ValueError(f"dataset_name doit être parmi {tuple(PATHS)}")

        base_root = Path(root) if root is not None else TEXT2HOI_ROOT
        self.root = base_root / self.dataset_name
        self.max_frames = int(max_frames)
        self.train = bool(train)
        self.use_paraphrases = bool(use_paraphrases)

        if self.max_frames <= 0:
            raise ValueError("max_frames doit être strictement positif")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        data_path = self.root / "data.npz"
        keys = (
            "x_lhand",
            "x_rhand",
            "j_lhand",
            "j_rhand",
            "x_obj",
            "lhand_beta",
            "rhand_beta",
            "lhand_org",
            "rhand_org",
            "is_lhand",
            "is_rhand",
            "action_name",
            "nframes",
        )
        self.data = {key: _npz_array(data_path, key) for key in keys}

        if self.dataset_name == "arctic":
            self.data["object_name"] = _npz_array(data_path, "object_name")
            self.data["x_obj_angle"] = _npz_array(data_path, "x_obj_angle")
        elif self.dataset_name == "grab":
            self.data["object_name"] = _npz_array(data_path, "obj_name")
            self.data["text_object_name"] = _npz_array(data_path, "proc_obj_name")
        else:
            self.data["object_idx"] = _npz_array(data_path, "object_idx")

        with open(self.root / "text.json", encoding="utf-8") as file:
            self.texts = json.load(file)

        with open(self.root / "obj.pkl", "rb") as file:
            object_data = pickle.load(file)

        self.object_names = object_data["object_name"]
        self.object_points = object_data["obj_pcs"]
        self.object_normals = object_data["obj_pc_normals"]

    def __len__(self):
        return len(self.data["nframes"])

    def _object_names(self, index):
        if self.dataset_name == "h2o":
            object_name = self.object_names[int(self.data["object_idx"][index])]
            return object_name, object_name
        object_name = str(self.data["object_name"][index])
        if self.dataset_name == "grab":
            return object_name, str(self.data["text_object_name"][index])
        return object_name, object_name

    def _text(self, index, is_lhand, is_rhand, text_object_name):
        action_name = str(self.data["action_name"][index])
        base = action_name if self.dataset_name == "h2o" else f"{action_name} {text_object_name}"
        if is_lhand and is_rhand:
            hand = "both hands"
        elif is_lhand:
            hand = "left hand"
        else:
            hand = "right hand"
        canonical_text = f"{base.capitalize()} with {hand}."
        variants = self.texts.get(canonical_text, [canonical_text])
        if self.train and self.use_paraphrases:
            text = variants[np.random.randint(len(variants))]
        else:
            text = variants[0]
        return canonical_text, text, action_name

    def _pad(self, key, index, start, length):
        values = self.data[key][index, start : start + length]
        output = np.zeros((self.max_frames, *values.shape[1:]), dtype=np.float32)
        output[:length] = values
        return torch.from_numpy(output)

    def __getitem__(self, index):
        total_frames = int(self.data["nframes"][index])
        length = min(total_frames, self.max_frames)
        if total_frames > self.max_frames:
            if self.train:
                start = np.random.randint(total_frames - self.max_frames + 1)
            else:
                start = (total_frames - self.max_frames) // 2
        else:
            start = 0

        is_lhand = bool(self.data["is_lhand"][index])
        is_rhand = bool(self.data["is_rhand"][index])
        object_name, text_object_name = self._object_names(index)
        canonical_text, text, action_name = self._text(
            index, is_lhand, is_rhand, text_object_name
        )

        x_lhand = self._pad("x_lhand", index, start, length)
        x_rhand = self._pad("x_rhand", index, start, length)
        j_lhand = self._pad("j_lhand", index, start, length)
        j_rhand = self._pad("j_rhand", index, start, length)
        lhand_beta = self._pad("lhand_beta", index, start, length)
        rhand_beta = self._pad("rhand_beta", index, start, length)
        lhand_org = self._pad("lhand_org", index, start, length)
        rhand_org = self._pad("rhand_org", index, start, length)

        if not is_lhand:
            x_lhand.zero_()
            j_lhand.zero_()
            lhand_beta.zero_()
            lhand_org.zero_()
        if not is_rhand:
            x_rhand.zero_()
            j_rhand.zero_()
            rhand_beta.zero_()
            rhand_org.zero_()

        frame_mask = torch.arange(self.max_frames) < length
        x_obj_angle = torch.zeros(self.max_frames, 1, dtype=torch.float32)
        if self.dataset_name == "arctic":
            x_obj_angle = self._pad("x_obj_angle", index, start, length)

        object_points = torch.as_tensor(
            np.asarray(self.object_points[object_name], dtype=np.float32).copy()
        )
        object_normals = torch.as_tensor(
            np.asarray(self.object_normals[object_name], dtype=np.float32).copy()
        )

        return {
            "x_lhand": x_lhand,
            "x_rhand": x_rhand,
            "j_lhand": j_lhand,
            "j_rhand": j_rhand,
            "x_obj": self._pad("x_obj", index, start, length),
            "x_obj_angle": x_obj_angle,
            "lhand_beta": lhand_beta,
            "rhand_beta": rhand_beta,
            "lhand_org": lhand_org,
            "rhand_org": rhand_org,
            "frame_mask": frame_mask,
            "lhand_mask": frame_mask & is_lhand,
            "rhand_mask": frame_mask & is_rhand,
            "is_lhand": torch.tensor(is_lhand),
            "is_rhand": torch.tensor(is_rhand),
            "nframes": torch.tensor(length, dtype=torch.long),
            "object_points": object_points,
            "object_normals": object_normals,
            "text": text,
            "canonical_text": canonical_text,
            "action_name": action_name,
            "object_name": text_object_name,
            "dataset_name": self.dataset_name,
            "index": torch.tensor(index, dtype=torch.long),
        }


def get_Text2HOI_loader(
    dataset_name="arctic",
    batch_size=32,
    shuffle=True,
    root=None,
    max_frames=150,
    train=True,
    use_paraphrases=True,
    num_workers=0,
    drop_last=False,
):
    dataset = Text2HOI(
        dataset_name=dataset_name,
        root=root,
        max_frames=max_frames,
        train=train,
        use_paraphrases=use_paraphrases,
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

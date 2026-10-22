import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Mano3DPlot:
    hand_chains = (
        (0, 13, 14, 15, 16),
        (0, 1, 2, 3, 17),
        (0, 4, 5, 6, 18),
        (0, 10, 11, 12, 19),
        (0, 7, 8, 9, 20),
    )
    hand_edges = tuple(
        (start, end)
        for chain in hand_chains
        for start, end in zip(chain, chain[1:])
    )

    def __init__(
        self,
        mano_model_dir=None,
        fps=20,
        figsize=(7, 7),
        left_color="#4C78A8",
        right_color="#E45756",
    ):
        project_root = Path(__file__).resolve().parents[1]
        self.mano_model_dir = Path(
            mano_model_dir
            or project_root / "datasets" / "mano_v1_2" / "models"
        )
        self.fps = fps
        self.figsize = figsize
        self.left_color = left_color
        self.right_color = right_color

    def animate_joints(
        self,
        joints,
        is_right=False,
        title="",
        max_frames=80,
        repeat=True,
    ):
        joints = self._prepare_joints(joints)
        return self._animate(
            hands=[
                self._hand(
                    joints=joints,
                    is_right=is_right,
                )
            ],
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def animate_hands(
        self,
        left_joints=None,
        right_joints=None,
        title="",
        max_frames=80,
        repeat=True,
    ):
        hands = []
        if left_joints is not None:
            hands.append(
                self._hand(
                    joints=self._prepare_joints(left_joints),
                    is_right=False,
                )
            )
        if right_joints is not None:
            hands.append(
                self._hand(
                    joints=self._prepare_joints(right_joints),
                    is_right=True,
                )
            )
        if not hands:
            raise ValueError("Au moins une main doit être fournie")

        return self._animate(
            hands=hands,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def animate_mano(
        self,
        parameters,
        betas,
        is_right=False,
        joints=None,
        show_joints=False,
        title="",
        max_frames=80,
        repeat=True,
        mesh_alpha=0.85,
    ):
        parameters = self._to_numpy(parameters)
        vertices, faces = self._decode_mano(parameters, betas, is_right)

        if joints is not None:
            joints = self._prepare_joints(joints)
            if len(joints) != len(vertices):
                raise ValueError("joints et parameters doivent avoir le même nombre de frames")
        elif show_joints:
            raise ValueError("joints est requis lorsque show_joints=True")

        return self._animate(
            hands=[
                self._hand(
                    joints=joints if show_joints else None,
                    vertices=vertices,
                    faces=faces,
                    is_right=is_right,
                    mesh_alpha=mesh_alpha,
                )
            ],
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def animate_mano_hands(
        self,
        left_parameters=None,
        left_betas=None,
        right_parameters=None,
        right_betas=None,
        left_joints=None,
        right_joints=None,
        show_joints=False,
        title="",
        max_frames=80,
        repeat=True,
        mesh_alpha=0.85,
    ):
        hands = []
        entries = (
            (left_parameters, left_betas, left_joints, False),
            (right_parameters, right_betas, right_joints, True),
        )

        for parameters, betas, joints, is_right in entries:
            if parameters is None:
                continue
            if betas is None:
                raise ValueError("betas est requis avec parameters")

            parameters = self._to_numpy(parameters)
            vertices, faces = self._decode_mano(parameters, betas, is_right)
            if joints is not None:
                joints = self._prepare_joints(joints)
                if len(joints) != len(vertices):
                    raise ValueError(
                        "joints et parameters doivent avoir le même nombre de frames"
                    )
            elif show_joints:
                raise ValueError("joints est requis lorsque show_joints=True")

            hands.append(
                self._hand(
                    joints=joints if show_joints else None,
                    vertices=vertices,
                    faces=faces,
                    is_right=is_right,
                    mesh_alpha=mesh_alpha,
                )
            )

        if not hands:
            raise ValueError("Au moins une main MANO doit être fournie")

        return self._animate(
            hands=hands,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def animate_handx_mano_hands(
        self,
        left_pose=None,
        left_betas=None,
        left_trans=None,
        right_pose=None,
        right_betas=None,
        right_trans=None,
        left_joints=None,
        right_joints=None,
        show_joints=False,
        title="",
        max_frames=80,
        repeat=True,
        mesh_alpha=0.85,
    ):
        hands = []
        entries = (
            (left_pose, left_betas, left_trans, left_joints, False),
            (right_pose, right_betas, right_trans, right_joints, True),
        )

        for pose, betas, translation, joints, is_right in entries:
            if pose is None:
                continue
            if betas is None or translation is None:
                raise ValueError("betas et translation sont requis avec pose")

            vertices, faces = self._decode_handx_mano(
                pose,
                betas,
                translation,
                is_right,
            )
            if joints is not None:
                joints = self._prepare_joints(joints)
                if len(joints) != len(vertices):
                    raise ValueError(
                        "joints et pose doivent avoir le même nombre de frames"
                    )
            elif show_joints:
                raise ValueError("joints est requis lorsque show_joints=True")

            hands.append(
                self._hand(
                    joints=joints if show_joints else None,
                    vertices=vertices,
                    faces=faces,
                    is_right=is_right,
                    mesh_alpha=mesh_alpha,
                )
            )

        if not hands:
            raise ValueError("Au moins une main MANO doit être fournie")

        return self._animate(
            hands=hands,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def _animate(
        self,
        hands,
        title,
        max_frames,
        repeat,
    ):
        frame_counts = [
            len(hand["vertices"])
            if hand["vertices"] is not None
            else len(hand["joints"])
            for hand in hands
        ]
        if len(set(frame_counts)) != 1:
            raise ValueError("Toutes les mains doivent avoir le même nombre de frames")

        frame_count = frame_counts[0]
        if max_frames is None:
            frame_ids = np.arange(frame_count)
        else:
            if max_frames <= 0:
                raise ValueError("max_frames doit être strictement positif")
            frame_ids = np.linspace(
                0,
                frame_count - 1,
                min(frame_count, max_frames),
                dtype=int,
            )

        figure = plt.figure(figsize=self.figsize)
        axis = figure.add_subplot(111, projection="3d")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.view_init(elev=20, azim=-60)

        limit_points = np.concatenate(
            [
                hand["vertices"].reshape(-1, 3)
                if hand["vertices"] is not None
                else hand["joints"].reshape(-1, 3)
                for hand in hands
            ],
            axis=0,
        )
        self._set_equal_limits(axis, limit_points)

        for hand in hands:
            color = hand["color"]
            hand["joint_artist"] = None
            hand["bone_artists"] = []
            hand["surface_artist"] = None

            if hand["joints"] is not None:
                hand["joint_artist"] = axis.plot(
                    [], [], [], "o", markersize=4, color=color
                )[0]
                hand["bone_artists"] = [
                    axis.plot([], [], [], color=color, linewidth=2.5)[0]
                    for _ in self.hand_edges
                ]

            if hand["vertices"] is not None:
                hand["surface_artist"] = Poly3DCollection(
                    hand["vertices"][0][hand["faces"]],
                    facecolor=color,
                    edgecolor="none",
                    alpha=hand["mesh_alpha"],
                )
                axis.add_collection3d(hand["surface_artist"])

        def update(animation_id):
            frame_id = int(frame_ids[animation_id])
            artists = []

            for hand in hands:
                if hand["joint_artist"] is not None:
                    frame_joints = hand["joints"][frame_id]
                    hand["joint_artist"].set_data_3d(
                        frame_joints[:, 0],
                        frame_joints[:, 1],
                        frame_joints[:, 2],
                    )
                    artists.append(hand["joint_artist"])
                    for artist, (start, end) in zip(
                        hand["bone_artists"], self.hand_edges
                    ):
                        segment = frame_joints[[start, end]]
                        artist.set_data_3d(
                            segment[:, 0],
                            segment[:, 1],
                            segment[:, 2],
                        )
                        artists.append(artist)

                if hand["surface_artist"] is not None:
                    hand["surface_artist"].set_verts(
                        hand["vertices"][frame_id][hand["faces"]]
                    )
                    artists.append(hand["surface_artist"])

            prefix = f"{title}\n" if title else ""
            axis.set_title(f"{prefix}frame {frame_id + 1}/{frame_count}")
            return artists

        update(0)
        return FuncAnimation(
            figure,
            update,
            frames=len(frame_ids),
            interval=1000 / self.fps,
            blit=False,
            repeat=repeat,
        )

    def _hand(
        self,
        joints=None,
        vertices=None,
        faces=None,
        is_right=False,
        mesh_alpha=0.85,
    ):
        return {
            "joints": joints,
            "vertices": vertices,
            "faces": faces,
            "color": self.right_color if is_right else self.left_color,
            "mesh_alpha": mesh_alpha,
        }

    def _decode_mano(self, parameters, betas, is_right):
        if parameters.ndim != 2 or parameters.shape[1] != 99:
            raise ValueError("parameters doit avoir la forme [T, 99]")

        model_path = self.mano_model_dir / (
            "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
        )
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        aliases = {
            "bool": np.bool_,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "str": str,
            "unicode": str,
        }
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)

        try:
            import chumpy
            import smplx
            import torch
            from scipy.spatial.transform import Rotation
        except ImportError as error:
            raise ImportError("La visualisation MANO requiert torch, scipy, smplx et chumpy") from error

        frame_count = len(parameters)
        rotation_matrices = self._rotation_6d_to_matrix(
            parameters[:, 3:].reshape(frame_count, 16, 6)
        )
        axis_angles = Rotation.from_matrix(
            rotation_matrices.reshape(-1, 3, 3)
        ).as_rotvec().reshape(frame_count, 16, 3)

        betas = self._to_numpy(betas)
        if betas.ndim == 1:
            betas = np.repeat(betas[None], frame_count, axis=0)
        if betas.shape != (frame_count, 10):
            raise ValueError("betas doit avoir la forme [10] ou [T, 10]")

        mano = smplx.MANO(
            str(self.mano_model_dir),
            is_rhand=is_right,
            use_pca=False,
            flat_hand_mean=False,
            create_transl=False,
            batch_size=frame_count,
        )
        axis_angles = torch.as_tensor(axis_angles, dtype=torch.float32)
        betas = torch.as_tensor(betas, dtype=torch.float32)

        with torch.no_grad():
            output = mano(
                betas=betas,
                global_orient=axis_angles[:, 0],
                hand_pose=axis_angles[:, 1:].reshape(frame_count, 45),
                return_verts=True,
            )

        vertices = output.vertices.cpu().numpy()
        vertices = vertices + parameters[:, None, :3]
        faces = np.asarray(mano.faces, dtype=np.int32)
        return vertices, faces

    def _decode_handx_mano(self, pose, betas, translation, is_right):
        pose = self._to_numpy(pose)
        betas = self._to_numpy(betas)
        translation = self._to_numpy(translation)

        if pose.ndim != 2 or pose.shape[1] != 48:
            raise ValueError("pose doit avoir la forme [T, 48]")

        frame_count = len(pose)
        if betas.ndim == 1:
            betas = np.repeat(betas[None], frame_count, axis=0)
        if betas.shape != (frame_count, 10):
            raise ValueError("betas doit avoir la forme [10] ou [T, 10]")
        if translation.shape != (frame_count, 3):
            raise ValueError("translation doit avoir la forme [T, 3]")

        model_path = self.mano_model_dir / (
            "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
        )
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec

        aliases = {
            "bool": np.bool_,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "str": str,
            "unicode": str,
        }
        for name, value in aliases.items():
            if name not in np.__dict__:
                setattr(np, name, value)

        try:
            import chumpy
            import smplx
            import torch
        except ImportError as error:
            raise ImportError("La visualisation MANO requiert torch, smplx et chumpy") from error

        mano = smplx.MANO(
            str(self.mano_model_dir),
            is_rhand=is_right,
            use_pca=False,
            flat_hand_mean=False,
            create_transl=False,
            batch_size=frame_count,
        )
        pose = torch.as_tensor(pose, dtype=torch.float32)
        betas = torch.as_tensor(betas, dtype=torch.float32)

        with torch.no_grad():
            output = mano(
                betas=betas,
                global_orient=pose[:, :3],
                hand_pose=pose[:, 3:],
                return_verts=True,
            )

        vertices = output.vertices.cpu().numpy()
        vertices = vertices + translation[:, None]
        faces = np.asarray(mano.faces, dtype=np.int32)
        return vertices, faces

    @staticmethod
    def _prepare_joints(joints):
        joints = Mano3DPlot._to_numpy(joints)
        if joints.ndim != 3 or joints.shape[1:] != (21, 3):
            raise ValueError("joints doit avoir la forme [T, 21, 3]")
        if len(joints) == 0:
            raise ValueError("joints ne doit pas être vide")
        return joints

    @staticmethod
    def _to_numpy(values):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _rotation_6d_to_matrix(rotation):
        rotation = np.asarray(rotation)
        output_shape = rotation.shape[:-1]
        values = rotation.reshape(-1, 3, 2)
        first = values[:, :, 0]
        second = values[:, :, 1]
        first = first / np.clip(
            np.linalg.norm(first, axis=-1, keepdims=True), 1e-8, None
        )
        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        second = second / np.clip(
            np.linalg.norm(second, axis=-1, keepdims=True), 1e-8, None
        )
        third = np.cross(first, second)
        matrices = np.stack((first, second, third), axis=-1)
        return matrices.reshape(*output_shape, 3, 3)

    @staticmethod
    def _set_equal_limits(axis, points):
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2
        radius = max((maximum - minimum).max() / 2, 1e-3)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))

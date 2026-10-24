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
        device="cpu",
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        project_root = Path(__file__).resolve().parents[1]
        self.mano_model_dir = Path(
            mano_model_dir
            or project_root / "datasets" / "mano_v1_2" / "models"
        )
        self.fps = fps
        self.figsize = figsize
        self.left_color = left_color
        self.right_color = right_color
        self.device = device
        self._mano_models = {}

    def save_gif(self, animation, path, dpi=100):
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        path = Path(path)
        if path.suffix.lower() != ".gif":
            raise ValueError("GIF path must use the .gif extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(
            str(path),
            writer="pillow",
            fps=self.fps,
            dpi=dpi,
        )
        return path

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
            raise ValueError("At least one hand must be provided")

        return self._animate(
            hands=hands,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
        )

    def animate_record(
        self,
        record,
        show_joints=False,
        title="",
        max_frames=80,
        repeat=True,
        mesh_alpha=0.85,
    ):
        return self.animate_features(
            features=record["x"],
            left_betas=record["left_shape"],
            right_betas=record["right_shape"],
            joint_positions=record.get("joint_positions"),
            show_joints=show_joints,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
            mesh_alpha=mesh_alpha,
        )

    def decode_features(self, features, left_betas, right_betas):
        from .mano import split_mano_6d

        features = self._to_numpy(features)
        if features.ndim != 2 or features.shape[1] != 198:
            raise ValueError("features must have shape [T, 198]")
        parts = split_mano_6d(features)
        left_vertices, left_faces = self._decode_mano(
            parts["left_parameters"],
            left_betas,
            False,
        )
        right_vertices, right_faces = self._decode_mano(
            parts["right_parameters"],
            right_betas,
            True,
        )
        return {
            "left_vertices": left_vertices,
            "left_faces": left_faces,
            "right_vertices": right_vertices,
            "right_faces": right_faces,
        }

    def animate_features(
        self,
        features,
        left_betas,
        right_betas,
        joint_positions=None,
        show_joints=False,
        title="",
        max_frames=80,
        repeat=True,
        mesh_alpha=0.85,
    ):
        from .mano import split_mano_6d

        features = self._to_numpy(features)
        if features.ndim != 2 or features.shape[1] != 198:
            raise ValueError("features must have shape [T, 198]")

        parts = split_mano_6d(features)
        left_joints = None
        right_joints = None
        if joint_positions is not None:
            joint_positions = self._to_numpy(joint_positions)
            if joint_positions.shape != (len(features), 2, 21, 3):
                raise ValueError("joint_positions must have shape [T, 2, 21, 3]")
            left_joints = joint_positions[:, 0]
            right_joints = joint_positions[:, 1]

        return self.animate_mano_hands(
            left_parameters=parts["left_parameters"],
            left_betas=left_betas,
            right_parameters=parts["right_parameters"],
            right_betas=right_betas,
            left_joints=left_joints,
            right_joints=right_joints,
            show_joints=show_joints,
            title=title,
            max_frames=max_frames,
            repeat=repeat,
            mesh_alpha=mesh_alpha,
        )

    def animate_feature_comparison(
        self,
        ground_truth,
        prediction,
        left_betas,
        right_betas,
        title="",
        max_frames=None,
        repeat=True,
        mesh_alpha=0.85,
    ):
        decoded = (
            (
                "Ground truth",
                self.decode_features(ground_truth, left_betas, right_betas),
            ),
            ("Prediction", self.decode_features(prediction, left_betas, right_betas)),
        )
        frame_count = len(decoded[0][1]["left_vertices"])
        if len(decoded[1][1]["left_vertices"]) != frame_count:
            raise ValueError("ground_truth and prediction must have the same length")
        if max_frames is None:
            frame_ids = np.arange(frame_count)
        else:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
            frame_ids = np.linspace(
                0,
                frame_count - 1,
                min(frame_count, max_frames),
                dtype=int,
            )

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(2 * self.figsize[0], self.figsize[1]),
            subplot_kw={"projection": "3d"},
        )
        points = np.concatenate(
            [
                values[field].reshape(-1, 3)
                for _, values in decoded
                for field in ("left_vertices", "right_vertices")
            ],
            axis=0,
        )
        artists = []
        for axis, (name, values) in zip(axes, decoded):
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_zlabel("z")
            axis.view_init(elev=20, azim=-60)
            self._set_equal_limits(axis, points)
            collections = []
            for vertices, faces, color in (
                (
                    values["left_vertices"],
                    values["left_faces"],
                    self.left_color,
                ),
                (
                    values["right_vertices"],
                    values["right_faces"],
                    self.right_color,
                ),
            ):
                collection = Poly3DCollection(
                    vertices[0][faces],
                    facecolor=color,
                    edgecolor="none",
                    alpha=mesh_alpha,
                )
                axis.add_collection3d(collection)
                collections.append((collection, vertices, faces))
            artists.append((axis, name, collections))

        if title:
            figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))

        def update(animation_id):
            frame_id = int(frame_ids[animation_id])
            updated = []
            for axis, name, collections in artists:
                axis.set_title(f"{name} — frame {frame_id + 1}/{frame_count}")
                for collection, vertices, faces in collections:
                    collection.set_verts(vertices[frame_id][faces])
                    updated.append(collection)
            return updated

        update(0)
        return FuncAnimation(
            figure,
            update,
            frames=len(frame_ids),
            interval=1000 / self.fps,
            blit=False,
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
                raise ValueError("betas is required with parameters")

            parameters = self._to_numpy(parameters)
            vertices, faces = self._decode_mano(parameters, betas, is_right)
            if joints is not None:
                joints = self._prepare_joints(joints)
                if len(joints) != len(vertices):
                    raise ValueError(
                        "joints and parameters must have the same number of frames"
                    )
            elif show_joints:
                raise ValueError("joints is required when show_joints=True")

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
            raise ValueError("At least one MANO hand must be provided")

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
                raise ValueError("betas and translation are required with pose")

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
                        "joints and pose must have the same number of frames"
                    )
            elif show_joints:
                raise ValueError("joints is required when show_joints=True")

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
            raise ValueError("At least one MANO hand must be provided")

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
            raise ValueError("All hands must have the same number of frames")

        frame_count = frame_counts[0]
        if max_frames is None:
            frame_ids = np.arange(frame_count)
        else:
            if max_frames <= 0:
                raise ValueError("max_frames must be positive")
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
        parameters = self._to_numpy(parameters)
        if parameters.ndim != 2 or parameters.shape[1] != 99:
            raise ValueError("parameters must have shape [T, 99]")
        frame_count = len(parameters)
        if frame_count == 0:
            raise ValueError("parameters must not be empty")

        torch, _ = self._mano_dependencies()
        from scipy.spatial.transform import Rotation
        from .mano import rotation_6d_to_matrix

        rotations = torch.as_tensor(
            parameters[:, 3:].reshape(frame_count, 16, 6),
            dtype=torch.float32,
        )
        rotation_matrices = rotation_6d_to_matrix(rotations).cpu().numpy()
        axis_angles = Rotation.from_matrix(
            rotation_matrices.reshape(-1, 3, 3)
        ).as_rotvec().reshape(frame_count, 16, 3)
        return self._decode_axis_angle_mano(
            axis_angles.reshape(frame_count, 48),
            betas,
            parameters[:, :3],
            is_right,
        )

    def _decode_handx_mano(self, pose, betas, translation, is_right):
        return self._decode_axis_angle_mano(
            pose,
            betas,
            translation,
            is_right,
        )

    def _decode_axis_angle_mano(self, pose, betas, translation, is_right):
        pose = self._to_numpy(pose)
        betas = self._to_numpy(betas)
        translation = self._to_numpy(translation)

        if pose.ndim != 2 or pose.shape[1] != 48:
            raise ValueError("pose must have shape [T, 48]")

        frame_count = len(pose)
        if frame_count == 0:
            raise ValueError("pose must not be empty")
        if betas.ndim == 1:
            betas = np.repeat(betas[None], frame_count, axis=0)
        if betas.shape != (frame_count, 10):
            raise ValueError("betas must have shape [10] or [T, 10]")
        if translation.shape != (frame_count, 3):
            raise ValueError("translation must have shape [T, 3]")

        torch, smplx = self._mano_dependencies()
        mano = self._get_mano_model(is_right, frame_count, smplx)
        pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device)
        betas = torch.as_tensor(betas, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            output = mano(
                betas=betas,
                global_orient=pose[:, :3],
                hand_pose=pose[:, 3:],
                return_verts=True,
            )

        vertices = output.vertices.detach().cpu().numpy()
        vertices = vertices + translation[:, None]
        faces = np.asarray(mano.faces, dtype=np.int32)
        return vertices, faces

    def _get_mano_model(self, is_right, frame_count, smplx):
        key = (bool(is_right), int(frame_count), str(self.device))
        if key not in self._mano_models:
            model_path = self.mano_model_dir / (
                "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
            )
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            self._mano_models[key] = smplx.MANO(
                str(self.mano_model_dir),
                is_rhand=is_right,
                use_pca=False,
                flat_hand_mean=False,
                create_transl=False,
                batch_size=frame_count,
            ).to(self.device).eval()
        return self._mano_models[key]

    @staticmethod
    def _mano_dependencies():
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
            __import__("chumpy")
            import smplx
            import torch
        except ImportError as error:
            raise ImportError(
                "MANO visualization requires torch, smplx, and chumpy"
            ) from error
        return torch, smplx

    @staticmethod
    def _prepare_joints(joints):
        joints = Mano3DPlot._to_numpy(joints)
        if joints.ndim != 3 or joints.shape[1:] != (21, 3):
            raise ValueError("joints must have shape [T, 21, 3]")
        if len(joints) == 0:
            raise ValueError("joints must not be empty")
        return joints

    @staticmethod
    def _to_numpy(values):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float32)

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

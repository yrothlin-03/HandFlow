from importlib import import_module

from .handx_loader import HandX, collate_handx, get_handx_loaders
from .mano import (
    MANO_BIMANUAL_DIM,
    MANO_HAND_DIM,
    axis_angle_to_matrix,
    handx_to_mano_6d,
    matrix_to_rotation_6d,
    pose_axis_angle_to_rotation_6d,
    rotation_6d_to_matrix,
    split_mano_6d,
)


def __getattr__(name):
    if name == "Mano3DPlot":
        return import_module(".3d_mano_plot", __name__).Mano3DPlot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HandX",
    "MANO_BIMANUAL_DIM",
    "MANO_HAND_DIM",
    "Mano3DPlot",
    "collate_handx",
    "get_handx_loaders",
    "axis_angle_to_matrix",
    "handx_to_mano_6d",
    "matrix_to_rotation_6d",
    "pose_axis_angle_to_rotation_6d",
    "rotation_6d_to_matrix",
    "split_mano_6d",
]

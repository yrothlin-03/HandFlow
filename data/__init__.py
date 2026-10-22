from importlib import import_module

from .handx_loader import HandX, get_handx_loader
from .text2hoi_loader import Text2HOI, get_Text2HOI_loader


def __getattr__(name):
    if name == "Mano3DPlot":
        return import_module(".3d_mano_plot", __name__).Mano3DPlot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HandX",
    "Mano3DPlot",
    "Text2HOI",
    "get_Text2HOI_loader",
    "get_handx_loader",
]

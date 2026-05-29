"""Mesh I/O for Python."""

from . import aster, med, shp
from .__about__ import __version__
from .core import *
from .utils import *


__all__ = [x for x in dir() if not x.startswith("_")]  # type: ignore
__all__ += [
    "__version__",
]

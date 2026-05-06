from __future__ import annotations

from enum import IntEnum


class CellType(IntEnum):
    """Enumeration of cell types."""

    empty = 0
    vertex = 1
    line = 3
    triangle = 5
    polygon = 7
    quad = 9
    tetra = 10
    hexahedron = 12
    wedge = 13
    pyramid = 14
    line3 = 21
    triangle6 = 22
    quad8 = 23

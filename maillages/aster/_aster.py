from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core import CellType


if TYPE_CHECKING:
    import os
    from typing import Literal, Optional

    from .. import Mesh


def write(
    filename: str | os.PathLike,
    mesh: Mesh,
    title: Optional[str] = None,
) -> None:
    """
    Write an ASTER file.

    Parameters
    ----------
    filename : str | PathLike
        Output file name.
    mesh : maillages.Mesh
        Input mesh.
    title : str, optional
        Title to add at the top of the file.

    """
    with open(filename, "w") as f:
        # Title
        if title:
            f.write(f"TITRE\n{title[:80]}\nFINSF\n\n")

        # Points
        mask = np.ptp(mesh.points, axis=0) == 0.0

        if mask.any():
            axis = np.flatnonzero(mask)[0]
            points = np.delete(mesh.points, axis, 1)
            prefix = "COOR_2D"

        else:
            points = mesh.points
            prefix = "COOR_3D"

        f.write(f"{prefix} NBOBJ={len(points)}\n")

        for i, point in enumerate(points):
            f.write(f"N{i} {' '.join(map(str, point))}\n")

        f.write("FINSF\n\n")

        # Cells
        celltype = None

        for i, (cell, celltype_) in enumerate(zip(mesh.cells, mesh.celltypes)):
            if celltype_ != celltype:
                if i > 0:
                    f.write("FINSF\n\n")

                f.write(f"{_maillages_to_aster_celltype[celltype_]}\n")

            f.write(f"M{i} {' '.join(map(lambda x: f'N{x}', cell))}\n")
            celltype = celltype_

        f.write("FINSF\n\n")

        # Point and cell groups
        write_group(f, mesh, "point")
        write_group(f, mesh, "cell")

        f.write("FIN\n")


def write_group(f, mesh: Mesh, entity: Literal["point", "cell"], n: int = 8) -> None:
    """Write point or cell groups, if any."""
    if entity == "point":
        prefix = "GROUP_NO"
        identifier = "N"
        data, tag_to_id = mesh._get_entity_tags("point")

    else:
        prefix = "GROUP_MA"
        identifier = "M"
        data, tag_to_id = mesh._get_entity_tags("cell")

    if data is None or tag_to_id is None:
        return

    for k, v in tag_to_id.items():
        mask = data == v if data.dtype.kind == "i" else data == k

        if mask.any():
            ids = np.flatnonzero(mask)
            ids = np.split(
                ids,
                np.arange(ids.size // n + int(ids.size % n != 0))[1:] * n,
            )

            f.write(f"{prefix} NOM={k}\n")
            for ids_ in ids:
                f.write(f"{' '.join(map(lambda x: f'{identifier}{x}', ids_))}\n")

            f.write("FINSF\n\n")


_maillages_to_aster_celltype = {
    CellType.vertex: "POI1",
    CellType.line: "SEG2",
    CellType.line3: "SEG3",
    CellType.triangle: "TRIA3",
    CellType.triangle6: "TRIA6",
    CellType.quad: "QUAD4",
    CellType.quad8: "QUAD8",
    CellType.tetra: "TETRA4",
    CellType.hexahedron: "HEXA8",
    CellType.wedge: "PENTA6",
    CellType.pyramid: "PYRAM5",
}
_aster_to_maillages_celltype = {v: k for k, v in _maillages_to_aster_celltype.items()}

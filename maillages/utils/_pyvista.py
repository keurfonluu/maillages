from __future__ import annotations

from typing import TYPE_CHECKING

from pyrequire import require_package


if TYPE_CHECKING:
    from .. import Mesh

    import pyvista as pv


@require_package("pyvista")
def to_pyvista(mesh: Mesh) -> pv.UnstructuredGrid:
    """
    Convert a mesh to a PyVista grid.

    Parameters
    ----------
    mesh : maillages.Mesh
        Input mesh.

    Returns
    -------
    pyvista.UnstructuredGrid
        Output PyVista grid.
    
    """
    import pyvista as pv

    cells = []
    for cell in mesh.cells:
        cells += [len(cell), *cell]

    ugrid = pv.UnstructuredGrid(
        cells,
        mesh.celltypes,
        mesh.points,
    )
    
    for k, v in mesh.point_data.items():
        ugrid.point_data[k] = v

    for k, v in mesh.cell_data.items():
        ugrid.cell_data[k] = v

    return ugrid

from __future__ import annotations

from typing import TYPE_CHECKING

from pyrequire import require_package


if TYPE_CHECKING:
    import pyvista as pv

    from .. import Mesh


@require_package("pvgridder")
def from_pyvista(mesh: pv.DataObject | pv.DataSet) -> Mesh:
    """
    Convert a PyVista grid to a mesh.

    Parameters
    ----------
    mesh : pyvista.DataObject | pyvista.DataSet
        Input PyVista grid.

    Returns
    -------
    maillages.Mesh
        Output mesh.

    """
    from pvgridder import get_cell_connectivity

    from .. import Mesh

    cells = get_cell_connectivity(mesh, flatten=False)  # type: ignore

    return Mesh(
        mesh.points,
        cells,
        mesh.celltypes,
        point_data={k: v for k, v in mesh.point_data.items()},
        cell_data={k: v for k, v in mesh.cell_data.items()},
        metadata=dict(mesh.user_dict),
    )


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

    ugrid.user_dict = mesh.metadata

    return ugrid

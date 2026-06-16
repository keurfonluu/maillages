from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
from pyrequire import require_package

from ._helpers import deserialize_dict, serialize_dict


if TYPE_CHECKING:
    import pyvista as pv

    from .. import Cell, Mesh


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
    user_dict = dict(mesh.user_dict)
    time_steps = user_dict.pop("maillages:time_steps", None)

    return Mesh(
        mesh.points,
        cells,
        mesh.celltypes,
        point_data={k: v for k, v in mesh.point_data.items()},
        cell_data={k: v for k, v in mesh.cell_data.items()},
        time_steps=time_steps,
        metadata=deserialize_dict(dict(user_dict)),
    )


@overload
def to_pyvista(mesh: Mesh) -> pv.UnstructuredGrid: ...


@overload
def to_pyvista(mesh: Cell) -> pv.Cell: ...


@require_package("pyvista")
def to_pyvista(mesh: Mesh | Cell) -> pv.UnstructuredGrid | pv.Cell:
    """
    Convert a mesh or cell to a PyVista grid or cell.

    Parameters
    ----------
    mesh : maillages.Mesh | maillages.Cell
        Input mesh or single cell.

    Returns
    -------
    pyvista.UnstructuredGrid | pyvista.Cell
        Output PyVista grid or cell.

    """
    import pyvista as pv

    from .. import Cell

    if isinstance(mesh, Cell):
        cells = [mesh.n_points, *range(mesh.n_points)]
        celltypes = [mesh.celltype]

    else:
        cells = []
        for cell in mesh.cells:
            cells += [len(cell), *cell]

        celltypes = mesh.celltypes

    ugrid = pv.UnstructuredGrid(
        cells,
        celltypes,
        mesh.points,
    )

    for k, v in mesh.point_data.items():
        ugrid.point_data[k] = np.atleast_1d(v)

    for k, v in mesh.cell_data.items():
        ugrid.cell_data[k] = np.atleast_1d(v)

    ugrid.user_dict = serialize_dict(mesh.metadata)

    if mesh.time_steps is not None and len(mesh.time_steps) > 0:
        ugrid.user_dict["maillages:time_steps"] = mesh.time_steps.tolist()

    return ugrid.get_cell(0) if isinstance(mesh, Cell) else ugrid
